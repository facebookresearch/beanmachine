# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

Implements inverse autoregressive flows.

reference:

Germain, Mathieu, et al. "Made: Masked autoencoder for distribution estimation."
International Conference on Machine Learning. 2015.
http://proceedings.mlr.press/v37/germain15.pdf
(MADE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

MIT License, this work refers to an open source from Github, you can find the original code here:
https://github.com/karpathy/pytorch-normalizing-flows/blob/b60e119b37be10ce2930ef9fa17e58686aaf2b3d/nflib/made.py#L1
https://github.com/karpathy/pytorch-normalizing-flows/blob/b60e119b37be10ce2930ef9fa17e58686aaf2b3d/nflib/flows.py#L169



"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class InverseAutoregressiveLayer(nn.Module):
    """
    Inverse Autoregressive Flow that uses a MADE-style network for fast forward.

    An implementation of the bijective transform of Inverse Autoregressive Flow
    (IAF), using by default Eq (10) from Kingma Et Al., 2016,
    :math:`f{z} = mu_t + sigma_t dot f{x}` Eq(10)

    This variant of IAF is claimed by the authors to be more numerically stable
    than one using Eq (10),which is:
    :math:`f{y} = sigma_t dot f{x} + (1-sigma_t) dot mu_t` Eq(14)
    where :math:`sigma_t` is restricted to :math:`(0,1)`.

    Here we define a "stable" version of IAF based on Eq(14), which is same as
    ~pyro.distributions.affine_autoregressive.py. If the stable keyword argument
    is set to True then the transformation used in Eq(14).

    """

    def __init__(
        self,
        network_architecture,
        idx: int,
        stable: bool,
        dim: int,
        loga_min_clip: int = -5,
        loga_max_clip: int = 3,
    ):

        """
        Define the MaskedAutoencoder and set the clamp min and max to avoid
        the extreme value of sigma.

        :param network_architecture: it is a parameter class that users give to
        define a masked_autoencoder network,including: input_layer, output_layer,
        activation function, hidden_layer, and n_block.
        : param idx: the index of each IAF layer, starts from 0.
        : param stable: chose to use the "stable" version of IAF or not.
        : param dim: the size of the input layer of masked_autoencoder NN.

        """

        super().__init__()
        self.dim_ = dim
        # eg: network_architecture = MaskedAutoencoder(dim, dim*2, nn.ELU(),
        # hidden_layer = 30, n_block = 4, seed_num = 11)
        self.network_architecture_ = network_architecture
        self.idx_ = idx
        self.loga_min_clip_ = loga_min_clip
        self.loga_max_clip_ = loga_max_clip
        self.stable_ = stable

    def get_parameter(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        """
        Learn the parameter mu and loga from maskedautoencoder
        network, and apply clamping to loga to avoid extreme
        loga that break the gradient.

        : param x: the samples drawn from the previous distribution
        :return: mu,loga.

        """

        mu, loga = self.network_architecture_(x).split(self.dim_, dim=1)
        loga = (
            loga
            + (
                loga.clamp(min=self.loga_min_clip_, max=self.loga_max_clip_) - loga
            ).detach()
        )
        return mu, loga

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        """
        the forward method that compute the f(z) = z*sigma+mu and
        log_jacobian for a single IAF layer.

        : param x: the samples drawn from the previous distribution
        which later would be transformed by the IAF layer.
        :return: f(z), log_jacobian.

        """

        mu, loga = self.get_parameter(x)
        if self.stable_:
            z = x * torch.exp(loga) + (1 - torch.exp(loga)) * mu
        else:
            z = x * torch.exp(loga) + mu
        if self.idx_ % 2 == 1:
            tmp = z.clone()
            if len(z[0]) > 1:
                z[:, 0], z[:, 1] = tmp[:, 1], tmp[:, 0]
        log_jacobian = torch.sum(loga, dim=1)
        return z, log_jacobian

    def backward(self, z: Tensor) -> Tuple[Tensor, Tensor]:

        """
        the backward method is the inverse direction to compute z =
        (f(z)-mu)/sigma, and inverse log jacobian.

        : param z: the samples have been transformed by IAF layer.
        :returns: z = (f(z)-mu)/sigma, and inverse log jacobian. But
        we do not need to use the output.

        """

        x = torch.zeros(z.size())
        log_jacobian = torch.zeros(len(z))
        if self.idx_ % 2 == 1:
            tmp = z.clone()
            z[:, 0], z[:, 1] = tmp[:, 1], tmp[:, 0]
        mu, loga = self.get_parameter(z.clone())
        if self.stable_:
            x = (z - (1 - torch.exp(loga) * mu)) * torch.exp(-loga)
        else:
            x = (z - mu) * torch.exp(-loga)
        log_jacobian -= torch.sum(loga, dim=1)
        return x, log_jacobian
