#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoderMLP(nn.Module):
    """
    Torch implementation of an autoencoder.

    The encoder and decoder are Multilayer Perceptrons defined by users.
    Users can also specify prevalence and content covariates, as well as target labels to guide the encoding and decoding (see forward method).
    """
    def __init__(
            self, 
            encoder_dims=[2000, 1024, 512, 20], 
            encoder_non_linear_activation='relu',
            encoder_bias=True,
            decoder_dims=[20, 1024, 2000],
            decoder_non_linear_activation=None,
            decoder_bias=False, 
            dropout=0.0
            ):
        super(AutoEncoderMLP, self).__init__()

        self.encoder_dims = encoder_dims
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.decoder_dims = decoder_dims
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.dropout = nn.Dropout(p=dropout)
        if encoder_non_linear_activation is not None:
            self.encoder_nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[encoder_non_linear_activation]
        if decoder_non_linear_activation is not None:
            self.decoder_nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[decoder_non_linear_activation]

        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encoder_dims[i], encoder_dims[i+1], bias = encoder_bias)
            for i in range(len(encoder_dims)-1)
        })

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decoder_dims[i], decoder_dims[i+1], bias = decoder_bias)
            for i in range(len(decoder_dims)-1)
        })

    def encode(self, x):
        """
        Encode the input.
        """
        hid = x
        for i, (_,layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder)-1 and self.encoder_non_linear_activation is not None:
                hid = self.encoder_nonlin(hid)
        return hid

    def decode(self, z):
        """
        Decode the input.
        """
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder)-1 and self.decoder_non_linear_activation is not None:
                hid = self.decoder_nonlin(hid)
        return hid

    def forward(self, x, prevalence_covariates, content_covariates, target_labels, to_simplex=True):
        """
        Call the encoder and decoder methods. 
        Returns the reconstructed input and the encoded input.
        """
        if target_labels is not None:
            x = torch.cat((x, target_labels), 1)
        if prevalence_covariates is not None:
            x = torch.cat((x, prevalence_covariates), 1)
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        if content_covariates is not None:
            theta_x = torch.cat((theta, content_covariates), 1)
        else:
            theta_x = theta    
        x_recon = self.decode(theta_x)
        if to_simplex:
            return x_recon, theta
        else:
            return x_recon, z
    

class AutoEncoderSAGE(nn.Module):
    def __init__(
            self, 
            encoder_dims=[2000, 1024, 512, 20], 
            encoder_non_linear_activation='relu',
            encoder_bias=True,
            dropout=0.0,
            bow_size=0,
            content_covariate_size=0,
            estimate_interactions=False,
            log_word_frequencies=None,
            l1_beta_reg = 0, 
            l1_beta_c_reg = 0, 
            l1_beta_ci_reg = 0
            ):
        super(AutoEncoderSAGE, self).__init__()

        self.encoder_dims = encoder_dims
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.dropout = nn.Dropout(p=dropout)
        if encoder_non_linear_activation is not None:
            self.encoder_nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[encoder_non_linear_activation]
        
        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encoder_dims[i], encoder_dims[i+1], bias = encoder_bias)
            for i in range(len(encoder_dims)-1)
        })

        self.vocab_size = bow_size
        self.n_topic = encoder_dims[-1]
        self.content_covariate_size = content_covariate_size
        self.estimate_interactions = estimate_interactions

        self.beta = nn.Linear(self.n_topic, self.vocab_size)  
        if log_word_frequencies is not None:
            with torch.no_grad():
                self.beta.bias.copy_(log_word_frequencies)
        if content_covariate_size != 0:
            self.beta_c = nn.Linear(content_covariate_size, self.vocab_size, bias=False)
            if estimate_interactions:
                self.beta_ci = nn.Linear(content_covariate_size*self.n_topic, self.vocab_size, bias=False) 

        self.l1_beta_reg = l1_beta_reg
        self.l1_beta_c_reg = l1_beta_c_reg
        self.l1_beta_ci_reg = l1_beta_ci_reg

    def encode(self, x):
        """
        Encode the input.
        """
        hid = x
        for i, (_,layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder)-1 and self.encoder_non_linear_activation is not None:
                hid = self.encoder_nonlin(hid)
        return hid
    
    def decode(self,z,content_covariates):
        eta = self.beta(z)
        if content_covariates is not None:
            covariate_size = content_covariates.shape[1]
            eta += self.beta_c(content_covariates)
            if self.estimate_interactions:
                theta_rsh = z.unsqueeze(2)
                tc_emb_rsh = content_covariates.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci(covar_interactions.reshape((batch_size, self.n_topic * covariate_size)))
        return eta
    
    def forward(self, x, prevalence_covariates, content_covariates, target_labels, to_simplex=True):
        if target_labels is not None:
            x = torch.cat((x, target_labels), 1)
        if prevalence_covariates is not None:
            x = torch.cat((x, prevalence_covariates), 1)
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        x_recon = self.decode(theta, content_covariates)
        if to_simplex:
            return x_recon, theta
        else:
            return x_recon, z

    def sparsity_loss(self, l1_beta, l1_beta_c, l1_beta_ci, device):

        l1_strengths_beta = torch.from_numpy(l1_beta).to(device)
        beta_weights_sq = torch.pow(self.beta.weight, 2)
        sparsity_loss = self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.content_covariate_size != 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(device)
            beta_c_weights_sq = torch.pow(self.beta_c.weight, 2)
            sparsity_loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

            if self.estimate_interactions:
                l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(device)
                beta_ci_weights_sq = torch.pow(self.beta_ci.weight, 2)
                sparsity_loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum() 

        return sparsity_loss
    
    def update_jeffreys_priors(self, n_train, min_weights_sq=1e-6):

        l1_beta = None
        l1_beta_c = None
        l1_beta_ci = None

        weights = self.beta.weight.detach().cpu().numpy()
        weights_sq = weights ** 2
        weights_sq[weights_sq < min_weights_sq] = min_weights_sq
        l1_beta = 0.5 / weights_sq / float(n_train)

        if self.content_covariate_size != 0:
            weights = self.beta_c.weight.detach().cpu().numpy()
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_c = 0.5 / weights_sq / float(n_train)

            if self.estimate_interactions:
                weights = self.beta_ci.weight.detach().cpu().numpy()
                weights_sq = weights ** 2
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta_ci = 0.5 / weights_sq / float(n_train)

        return l1_beta, l1_beta_c, l1_beta_ci
    