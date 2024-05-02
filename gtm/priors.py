#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
from sklearn import linear_model
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from gtm.utils import compute_dirichlet_likelihood

class Prior:
    """
    Base template class for doc-topic priors.
    """
    def __init__(self):
        pass
    
    def update_parameters(self):
        """
        M-step after each epoch.
        """        
        pass
    
    def sample(self):
        """
        Sample from the prior.
        """
        pass

    def simulate(self):
        """
        Simulate data to test the prior's updating rule.
        """ 
        pass

    
class LogisticNormalPrior(Prior):
    """
    Logistic Normal prior
    
    We draw from a multivariate gaussian and map it to the simplex.
    Does not induce sparsity, but may account for topic correlations.
    
    References:
        - Roberts, M. E., Stewart, B. M., & Airoldi, E. M. (2016). A model of text for experimentation in the social sciences. Journal of the American Statistical Association, 111(515), 988-1003.
    """
    def __init__(self, prevalence_covariate_size, n_topics, prevalence_covariates_regularization, device):
        self.prevalence_covariates_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.prevalence_covariates_regularization = prevalence_covariates_regularization
        self.device = device
        if prevalence_covariate_size != 0:
            self.lambda_ = torch.zeros(prevalence_covariate_size, n_topics).to(self.device)
            self.sigma = torch.diag(torch.Tensor([1.0]*self.n_topics)).to(self.device)            
    
    def update_parameters(self, posterior_mu, M_prevalence_covariates): 
        """
        M-step after each epoch.
        """

        reg = linear_model.Ridge(alpha=self.prevalence_covariates_regularization, fit_intercept=False)
        lambda_ = reg.fit(M_prevalence_covariates, posterior_mu).coef_
        self.lambda_ = torch.from_numpy(lambda_.T).to(self.device)

        posterior_mu = torch.from_numpy(posterior_mu).to(self.device)
        M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(self.device)
        difference_in_means = posterior_mu - torch.matmul(M_prevalence_covariates, self.lambda_)
        self.sigma =  torch.matmul(difference_in_means.T, difference_in_means)/posterior_mu.shape[0] 

        self.lambda_ = self.lambda_ - self.lambda_[:,0][:,None]

    def sample(self, N, M_prevalence_covariates, to_simplex=True, epoch=None):
        """
        Sample from the prior.
        """
        if self.prevalence_covariates_size == 0:
            z_true = np.random.randn(N, self.n_topics)
        else:
            if torch.is_tensor(M_prevalence_covariates) == False:
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(self.device)
            means = torch.matmul(M_prevalence_covariates, self.lambda_)
            for i in range(means.shape[0]):
                if i == 0:
                    m = MultivariateNormal(means[i], self.sigma)
                    z_true = m.sample().unsqueeze(0)
                else:
                    m = MultivariateNormal(means[i], self.sigma)
                    z_temp = m.sample()
                    z_true = torch.cat([z_true, z_temp.unsqueeze(0)],0)
        if to_simplex:
            z_true = torch.softmax(z_true, dim=1)
        return z_true.float()
    
    def simulate(self, M_prevalence_covariates, lambda_, sigma, to_simplex=False):
        """
        Simulate data to test the prior's updating rule.
        """    
        means = torch.matmul(M_prevalence_covariates, lambda_)
        for i in range(means.shape[0]):
            if i == 0:
                m = MultivariateNormal(means[i], sigma)
                z_sim = m.sample().unsqueeze(0)
            else:
                m = MultivariateNormal(means[i], sigma)
                z_temp = m.sample()
                z_sim = torch.cat([z_sim, z_temp.unsqueeze(0)],0)
        if to_simplex:
            z_sim = torch.softmax(z_sim, dim=1)
        return z_sim.float()


class LinearModel(torch.nn.Module):
    """
    Simple linear model for the priors.
    """
    def __init__(self, prevalence_covariates_size, n_topics):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(prevalence_covariates_size, n_topics)
    
    def forward(self, M_prevalence_covariates):
        linear_preds = self.linear(M_prevalence_covariates)
        return linear_preds
    

class DirichletPrior(Prior):
    """
    Dirichlet prior

    Induces sparsity, but does not account for topic correlations.
    
    References:
        - Mimno, D. M., & McCallum, A. (2008, July). Topic models conditioned on arbitrary features with Dirichlet-multinomial regression. In UAI (Vol. 24, pp. 411-418).
        - Maier, M. (2014). DirichletReg: Dirichlet regression for compositional data in R.
    """
    def __init__(self, prevalence_covariates_size, n_topics, alpha, prevalence_covariates_regularization, tol, device):
        self.prevalence_covariates_size = prevalence_covariates_size
        self.n_topics = n_topics
        self.alpha = alpha
        self.prevalence_covariates_regularization = prevalence_covariates_regularization
        self.tol = tol
        self.lambda_ = None
        self.device = device
        if prevalence_covariates_size != 0:
            self.linear_model = LinearModel(prevalence_covariates_size, n_topics).to(self.device)
        
    def update_parameters(self, posterior_theta, M_prevalence_covariates, MLE=True):
        """
        M-step after each epoch.
        """

        # Simple Conditional Means Estimation
        # (fast educated guess)
        y = np.log(posterior_theta + 1e-6)
        reg = linear_model.Ridge(alpha=self.prevalence_covariates_regularization, fit_intercept=False)
        self.lambda_ = reg.fit(M_prevalence_covariates, y).coef_.T
        self.lambda_ = self.lambda_ - self.lambda_[:,0][:,None] 

        # Maximum Likelihood Estimation (MLE)
        # (pretty slow)
        self.lambda_ = torch.from_numpy(self.lambda_).float().to(self.device)
        posterior_theta = torch.from_numpy(posterior_theta).float().to(self.device)
        M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).float().to(self.device)
        with torch.no_grad():
            self.linear_model.linear.weight.copy_(self.lambda_.T)
        optimizer = torch.optim.Adam(self.linear_model.parameters(), lr=1e-3, weight_decay=self.prevalence_covariates_regularization)

        previous_loss = 0
        if MLE:
            while True:
                
                optimizer.zero_grad()
                linear_preds = self.linear_model(M_prevalence_covariates)
                alphas = torch.exp(linear_preds)
                loss = -compute_dirichlet_likelihood(alphas, posterior_theta)
                loss.backward()
                optimizer.step()

                if torch.abs(loss-previous_loss) < self.tol:
                    break 
                
                previous_loss = loss
                
                self.lambda_ = self.linear_model.linear.weight.detach().T
                self.lambda_ = self.lambda_ - self.lambda_[:,0][:,None]
                self.lambda_ = self.lambda_.cpu().numpy()
    
    def sample(self, N, M_prevalence_covariates, epoch=10):
        """
        Sample from the prior.
        """
        if self.prevalence_covariates_size == 0 or epoch == 0:
            z_true = np.random.dirichlet(np.ones(self.n_topics)*self.alpha, size=N)
            z_true = torch.from_numpy(z_true).float()
        else:
            with torch.no_grad():
                if torch.is_tensor(M_prevalence_covariates) == False:
                    M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(self.device)
                linear_preds = self.linear_model(M_prevalence_covariates)
                alphas = torch.exp(linear_preds)
                for i in range(alphas.shape[0]):
                    if i == 0:
                        d = Dirichlet(alphas[i])
                        z_true = d.sample().unsqueeze(0)
                    else:
                        d = Dirichlet(alphas[i])                    
                        z_temp = d.sample()
                        z_true = torch.cat([z_true, z_temp.unsqueeze(0)],0)
        return z_true
    
    def simulate(self, M_prevalence_covariates, lambda_):
        """
        Simulate data to test the prior's updating rule.
        """     
        alphas = torch.exp(torch.matmul(M_prevalence_covariates, lambda_))
        for i in range(alphas.shape[0]):
            if i == 0:
                d = Dirichlet(alphas[i])
                z_sim = d.sample().unsqueeze(0)
            else:
                d = Dirichlet(alphas[i])                    
                z_temp = d.sample()
                z_sim = torch.cat([z_sim, z_temp.unsqueeze(0)],0)
        return z_sim