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
      f'enc_{i}': nn.Linear(encoder_dims[i], encoder_dims[i + 1], bias=encoder_bias)
      for i in range(len(encoder_dims) - 1)
    })

    self.decoder = nn.ModuleDict({
      f'dec_{i}': nn.Linear(decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias)
      for i in range(len(decoder_dims) - 1)
    })
    # create another decoder for the second language
    self.decoder2 = nn.ModuleDict({
      f'dec2_{i}': nn.Linear(decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias)
      for i in range(len(decoder_dims) - 1)
    })

  def encode(self, x):
    """
    Encode the input.
    """
    hid = x
    for i, (_, layer) in enumerate(self.encoder.items()):
      hid = self.dropout(layer(hid))
      if i < len(self.encoder) - 1 and self.encoder_non_linear_activation is not None:
        hid = self.encoder_nonlin(hid)
    return hid

  def decode(self, z):
    """
    Decode the input.
    """
    hid = z
    for i, (_, layer) in enumerate(self.decoder.items()):
      hid = layer(hid)
      if i < len(self.decoder) - 1 and self.decoder_non_linear_activation is not None:
        hid = self.decoder_nonlin(hid)
    return hid

  def decode2(self, z):
    """
    Decode the input.
    """
    hid = z
    for i, (_, layer) in enumerate(self.decoder2.items()):
      hid = layer(hid)
      if i < len(self.decoder2) - 1 and self.decoder_non_linear_activation is not None:
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
    x_recon2 = self.decode2(theta_x)
    if to_simplex:
      return x_recon, x_recon2, theta
    else:
      return x_recon, x_recon2, z
