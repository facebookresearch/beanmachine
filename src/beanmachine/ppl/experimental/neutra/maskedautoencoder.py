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
from typing import Any, Callable

import torch
import torch.nn as nn
from beanmachine.ppl.experimental.neutra.maskedlinear import MaskedLinear
from torch import Tensor


class MaskedAutoencoder(nn.Module):

    """
    An implementation of the Masked Autoencoder (MADE: Masked Autoencoder
    for Distribution Estimation Germain, Mathieu, et al. 2015)

    Based on the paper, MADE would apply mask matrix to weight, so that we
    can contol the connect between nodes on each layer, to make them conditionally
    connected instead of fully connected. For a single hidden layer autoencoder,
    we write :math: h(x) = g(b + (W ⊙ M^W)x)
    So in the implementation, we build mask matrixs with 0,and 1. 0 means it is
    unconnected between nodes of two layers, 1 means connected.for the hidden layer,
    the node must be connected to the node not larger than it, But for the output
    layer, the node must be connected to the node smaller than it. So we need to
    consider them separately.And for the autoregressive property, the output layer
    would connect to the input layer.

    """

    def __init__(
        self,
        in_layer: int,
        out_layer: int,
        activation_function: Callable,
        hidden_layer: int,
        n_block: int,
        seed_num: int,
    ):
        """
        Initialize object.

        :param in_layer: the size of each input layers
        :param out_layer: the size of each output layers, must be k time of
        in_layer, k is int.
        :param activation_function :  nn.ReLU() or nn.ELU().
        :param hidden_layer: the size of each hidden layers, must be not
        samller than input_layer, default = 30
        :param n_block: how many hidden layers, default =  4
        """

        super().__init__()
        self.activation_function_ = activation_function
        self.masked_autoencoder = []
        self.permute_ = {}
        g1 = torch.Generator()
        # pyre-fixme
        g1.manual_seed(seed_num)  # assign a seed to generator
        self.create_masks_(in_layer, out_layer, n_block, hidden_layer, g1)

    def create_masks_(  # noqa: C901
        self, in_layer: int, out_layer: int, n_block: int, hidden_layer: int, g1: Any
    ) -> None:
        """
        Build the mask matrix, and set mask for each layer.
        :param in_layer: the size of each input layers
        :param out_layer: the size of each output layers, must be k time of
        in_layer, k is int.
        :param n_block: how many hidden layers, default =  4
        :param hidden_layer: the size of each hidden layers, must be not
        samller than input_layer, default = 30

        :returns: Nothing.

        """
        # check the input dimension of each layer is correct.
        if hidden_layer < in_layer:
            raise ValueError("Hidden dimension must not be less than input dimension.")
        if out_layer % in_layer != 0:
            raise ValueError("Output dimension must be k times of input dimension.")

        # Get the permutation, so we can track each node on different layers.
        self.permute_[-1] = torch.arange(
            in_layer
        )  # output layer would considered later for auto-regressive property.
        for layer in range(n_block):
            if in_layer > 1:
                # pyre-fixme
                self.permute_[layer] = torch.randint(
                    self.permute_[layer - 1].min(),
                    in_layer - 1,
                    (hidden_layer,),
                    generator=g1,
                )
            else:
                # pyre-fixme
                self.permute_[layer] = torch.randint(
                    self.permute_[layer - 1].min(), 2, (hidden_layer,), generator=g1
                )

        # Build mask matrix, procedure mask for each layer.
        # 1) Build hidden layer masks
        mask_ = []
        for layer in range(n_block):
            l1 = self.permute_[layer - 1]
            l2 = self.permute_[layer]
            mask_.append(torch.zeros([len(l1), len(l2)]))
            for x in range(len(l1)):
                for y in range(len(l2)):
                    # for the hidden layer, the conditional dependency is we
                    # only connect the node from the current layer to the node
                    # whose index is not larger than this node's index from the
                    # previous layer.
                    mask_[layer][x][y] = l1[x] <= l2[y]

        # 2) Build a mask to connect between input/output
        mask_.append(
            torch.zeros([len(self.permute_[n_block - 1]), len(self.permute_[-1])])
        )
        for x in range(len(self.permute_[n_block - 1])):
            for y in range(len(self.permute_[-1])):
                # for the output layer, the conditional dependency is we only
                # connect the node from the current layer to the node whose
                # index is smaller than this node's index from the previous
                # layer. This is different from the hidden layer rule.
                mask_[-1][x][y] = self.permute_[n_block - 1][x] < self.permute_[-1][y]
        k = out_layer // in_layer
        mask_[-1] = torch.cat([mask_[-1]] * k, dim=1)
        if in_layer == 1:
            for i, m_ in enumerate(mask_):
                m_ = torch.ones(m_.size())
                mask_[i] = m_
        # Create masked layers
        layers = [in_layer]
        for _ in range(n_block):
            layers.append(hidden_layer)
        layers.append(out_layer)
        # Build Network
        for i in range(1, len(layers)):
            layer_ = MaskedLinear(layers[i - 1], layers[i])
            layer_.set_mask(mask_[i - 1])
            self.masked_autoencoder.extend([layer_, self.activation_function_])
        self.masked_autoencoder.pop()
        self.masked_autoencoder = nn.Sequential(*self.masked_autoencoder)

    def forward(self, x: Tensor) -> Tensor:
        """
        the forward method that goes through the MaskedAutoencoder network,
        does computation and returns the network.
        :param x: how many hidden layers, default =  4
        :return: masked_autoencoder network
        """

        return self.masked_autoencoder(x)
