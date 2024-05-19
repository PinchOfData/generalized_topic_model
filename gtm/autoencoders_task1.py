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
    self.decoder_dims = decoder_dims

    encoder_layers = []
    for i in range(len(encoder_dims) - 1):
      encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1], bias=encoder_bias))
      encoder_layers.append(nn.Dropout(dropout))
      if i < len(encoder_dims) - 2:
        if encoder_non_linear_activation == 'relu':
          encoder_layers.append(nn.ReLU())
        elif encoder_non_linear_activation == 'sigmoid':
          encoder_layers.append(nn.Sigmoid())
        else:
          raise ValueError(f'Unknown activation function {encoder_non_linear_activation}')
    self.encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []
    for i in range(len(decoder_dims) - 1):
      decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias))
      decoder_layers.append(nn.Dropout(dropout))
      if i < len(decoder_dims) - 2:
        if decoder_non_linear_activation == 'relu':
          decoder_layers.append(nn.ReLU())
        elif decoder_non_linear_activation == 'sigmoid':
          decoder_layers.append(nn.Sigmoid())
        else:
          raise ValueError(f'Unknown activation function {decoder_non_linear_activation}')
    self.decoder = nn.Sequential(*decoder_layers)

  def encode(self, x):
    """
    Encode the input.
    """
    return self.encoder(x)

  def decode(self, z):
    """
    Decode the input.
    """
    return self.decoder(z)

  def forward(self, x, prevalence_covariates, content_covariates, target_labels, to_simplex=True):
    """
    Call the encoder and decoder methods.
    Returns the reconstructed input and the encoded input.
    """
    # if target_labels is not None:
    #   x = torch.cat((x, target_labels), 1)
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
