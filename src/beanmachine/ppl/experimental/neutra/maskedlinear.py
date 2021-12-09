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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedLinear(nn.Linear):

    """
    A linear mapping with a given mask on the weights

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """
        MaskedLinear creates a Linear operator with an optional mask on the weights.

        :param in_features: the number of input features
        :param out_features: the number of output features
        :param bias: whether or not `MaskedLinear` should include a bias term. defaults to `False`
        """
        self.masked_weight = None

        if in_features <= 0:
            raise ValueError("input feature dimension must be larger than 0.")
        if out_features <= 0:
            raise ValueError("output feature dimension must be larger than 0.")

        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: Tensor) -> None:
        """
        copy the mask matrix value to self.mask
        :param mask: the mask to apply to the in_features x out_features weight matrix
        :return: Nothing
        """
        if mask.t().size() != self.mask.size():
            raise ValueError("Dimension mismatches between mask and layer.")
        self.mask.data.copy_(mask.t())

    def forward(self, input_: Tensor) -> Tensor:
        """
        the forward method that does the masked linear computation
        and returns the result.
        :param input_: the layer with dimension in_features x out_features.
        :return: the output of the linear layer with masked weights.

        """
        self.masked_weight = self.weight * self.mask

        return F.linear(input_, self.masked_weight, self.bias)
