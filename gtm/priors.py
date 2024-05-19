#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from sklearn import linear_model
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from gtm.utils import compute_dirichlet_likelihood
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV, MultiTaskElasticNetCV


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

  def __init__(
      self,
      prevalence_covariate_size,
      n_topics,
      model_type,
      prevalence_model_args,
      device,
  ):
    self.prevalence_covariates_size = prevalence_covariate_size
    self.n_topics = n_topics
    self.model_type = model_type
    self.prevalence_model_args = prevalence_model_args
    self.device = device
    if prevalence_covariate_size != 0:
      self.lambda_ = torch.zeros(prevalence_covariate_size, n_topics).to(
        self.device
      )
      self.sigma = torch.diag(torch.Tensor([1.0] * self.n_topics)).to(self.device)

  def update_parameters(self, posterior_mu, M_prevalence_covariates):
    """
        M-step after each epoch.
        """
    if self.model_type == "MultiTaskElasticNetCV":
      reg = MultiTaskElasticNetCV(fit_intercept=False, **self.prevalence_model_args)
    elif self.model_type == "MultiTaskLassoCV":
      reg = MultiTaskLassoCV(fit_intercept=False, **self.prevalence_model_args)
    elif self.model_type == "RidgeCV":
      reg = RidgeCV(fit_intercept=False, **self.prevalence_model_args)
    else:
      reg = LinearRegression(fit_intercept=False, **self.prevalence_model_args)

    reg.fit(M_prevalence_covariates, posterior_mu)
    lambda_ = reg.coef_
    self.lambda_ = torch.from_numpy(lambda_.T).to(self.device)

    posterior_mu = torch.from_numpy(posterior_mu).to(self.device)
    M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
      self.device
    )
    difference_in_means = posterior_mu - torch.matmul(
      M_prevalence_covariates, self.lambda_.to(torch.float32)
    )
    self.sigma = (
        torch.matmul(difference_in_means.T, difference_in_means)
        / posterior_mu.shape[0]
    )

    self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]
    self.lambda_ = self.lambda_.to(torch.float32)

  def sample(self, N, M_prevalence_covariates, to_simplex=True, epoch=None):
    """
        Sample from the prior.
        """
    if self.prevalence_covariates_size == 0:
      z_true = np.random.randn(N, self.n_topics)
      z_true = torch.from_numpy(z_true).to(
        self.device
      )
    else:
      if torch.is_tensor(M_prevalence_covariates) == False:
        M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
          self.device
        )
      means = torch.matmul(M_prevalence_covariates, self.lambda_)
      z_true = torch.empty((means.shape[0], self.sigma.shape[0]))
      for i in range(means.shape[0]):
        m = MultivariateNormal(means[i], self.sigma)
        z_true[i] = m.sample()
    if to_simplex:
      z_true = torch.softmax(z_true, dim=1)
    return z_true.float()

  def simulate(self, M_prevalence_covariates, lambda_, sigma, to_simplex=False):
    """
        Simulate data to test the prior's updating rule.
        """
    means = torch.matmul(M_prevalence_covariates, lambda_)
    z_sim = torch.empty((means.shape[0], sigma.shape[0]))
    for i in range(means.shape[0]):
      m = MultivariateNormal(means[i], sigma)
      z_sim[i] = m.sample()
    if to_simplex:
      z_sim = torch.softmax(z_sim, dim=1)
    return z_sim.float()

  def get_topic_correlations(self):
    """
        Plot correlations between topics for a logistic normal prior.
        """
    # Represent as a standard variance-covariance matrix
    # See https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
    sigma = pd.DataFrame(self.sigma.detach().cpu().numpy())
    mask = np.zeros_like(sigma, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sigma[mask] = np.nan
    p = (
      sigma.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
      .highlight_null(color="#f1f1f1")  # Color NaNs grey
      .format(precision=2)
    )
    return p

  def to(self, device):
    """
        Move the model to a different device.
        """
    self.device = device
    self.lambda_ = self.lambda_.to(device)
    self.sigma = self.sigma.to(device)


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

  def __init__(
      self,
      prevalence_covariates_size,
      n_topics,
      alpha,
      prevalence_model_args,
      tol,
      device,
  ):
    self.prevalence_covariates_size = prevalence_covariates_size
    self.n_topics = n_topics
    self.alpha = alpha
    if prevalence_model_args is None:
      self.prevalence_model_args = {"alphas": [0, 0.1, 1, 10]}
    else:
      self.prevalence_model_args = prevalence_model_args
    self.tol = tol
    self.lambda_ = None
    self.device = device
    if prevalence_covariates_size != 0:
      self.linear_model = LinearModel(prevalence_covariates_size, n_topics).to(
        self.device
      )

  def update_parameters(self, posterior_theta, M_prevalence_covariates):
    """
        M-step after each epoch.
        """

    alphas = self.prevalence_model_args["alphas"]

    # Simple Conditional Means Estimation
    # (fast educated guess)
    y = np.log(posterior_theta + 1e-6)
    reg = linear_model.LinearRegression(fit_intercept=False)
    self.lambda_ = reg.fit(M_prevalence_covariates, y).coef_.T
    self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]

    # Maximum Likelihood Estimation (MLE)
    # (pretty slow)
    self.lambda_ = torch.from_numpy(self.lambda_).float().to(self.device)
    posterior_theta = torch.from_numpy(posterior_theta).float().to(self.device)
    M_prevalence_covariates = (
      torch.from_numpy(M_prevalence_covariates).float().to(self.device)
    )
    with torch.no_grad():
      self.linear_model.linear.weight.copy_(self.lambda_.T)

    best_loss = float("inf")

    for alpha in alphas:
      optimizer = torch.optim.Adam(
        self.linear_model.parameters(),
        weight_decay=alpha,
      )

      previous_loss = 0
      while True:
        optimizer.zero_grad()
        linear_preds = self.linear_model(M_prevalence_covariates)
        alphas = torch.exp(linear_preds)
        loss = -compute_dirichlet_likelihood(alphas, posterior_theta)
        loss.backward()
        optimizer.step()

        if torch.abs(loss - previous_loss) < self.tol:
          break

        previous_loss = loss

      if loss < best_loss:
        best_loss = loss

        self.lambda_ = self.linear_model.linear.weight.detach().T
        self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]
        self.lambda_ = self.lambda_.cpu().numpy()

  def sample(self, N, M_prevalence_covariates, epoch=10):
    """
        Sample from the prior.
        """
    if self.prevalence_covariates_size == 0 or epoch == 0:
      z_true = np.random.dirichlet(np.ones(self.n_topics) * self.alpha, size=N)
      z_true = torch.from_numpy(z_true).float()
    else:
      with torch.no_grad():
        if torch.is_tensor(M_prevalence_covariates) == False:
          M_prevalence_covariates = torch.from_numpy(
            M_prevalence_covariates
          ).to(self.device)
        linear_preds = self.linear_model(M_prevalence_covariates)
        alphas = torch.exp(linear_preds)
        alphas = torch.exp(torch.matmul(M_prevalence_covariates, self.lambda_))
        z_true = torch.empty((alphas.shape[0], self.lambda_.shape[1]))
        for i in range(alphas.shape[0]):
          m = Dirichlet(alphas[i])
          z_true[i] = m.sample()
    return z_true

  def simulate(self, M_prevalence_covariates, lambda_):
    """
        Simulate data to test the prior's updating rule.
        """
    alphas = torch.exp(torch.matmul(M_prevalence_covariates, lambda_))
    z_sim = torch.empty((alphas.shape[0], lambda_.shape[1]))
    for i in range(alphas.shape[0]):
      m = Dirichlet(alphas[i])
      z_sim[i] = m.sample()
    return z_sim

  def to(self, device):
    """
        Move the model to a different device.
        """
    self.device = device
    self.linear_model = self.linear_model.to(device)
