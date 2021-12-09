# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
import torch.nn as nn
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from torch.autograd import Variable


class MaskedAutoencoderTest(unittest.TestCase):
    def test_connection_and_stability(self):

        hidden_layer = 20
        n_block = 2
        in_layer = 10
        out_layer = 2 * in_layer
        seed_num = 11

        x = torch.rand(1, in_layer)

        model = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        # run backpropagation for each dimension to compute what
        # is the node dependency between each layer.
        for k in range(out_layer):
            xtr = Variable(x, requires_grad=True)
            xtrhat = model(xtr)
            loss = xtrhat[0, k]
            loss.backward()

            dependency = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            dependency_idx = list(np.where(dependency)[0])
            isok = k % in_layer not in dependency_idx
            # check the correctness of nodes dependency
            self.assertTrue(isok)

        first_run = model.masked_autoencoder(x)

        # run forward again to check if there is any stochastic error

        for k in range(out_layer):
            xtr2 = Variable(x, requires_grad=True)
            xtrhat2 = model(xtr2)
            loss = xtrhat2[0, k]
            loss.backward()

        second_run = model.masked_autoencoder(x)
        # check if the forward results of two runs are same
        self.assertCountEqual(first_run.tolist(), second_run.tolist())

    def test_input_shape(self):

        hidden_layer = 20
        n_block = 2
        in_layer = 80
        out_layer = 10

        with self.assertRaisesRegex(
            ValueError, "Hidden dimension must not be less than input dimension."
        ):
            MaskedAutoencoder(in_layer, out_layer, nn.ELU(), hidden_layer, n_block, 11)

    def test_input_and_output_shape(self):
        hidden_layer = 20
        n_block = 2
        in_layer = 8
        out_layer = 33

        with self.assertRaisesRegex(
            ValueError, "Output dimension must be k times of input dimension."
        ):
            MaskedAutoencoder(in_layer, out_layer, nn.ELU(), hidden_layer, n_block, 11)
