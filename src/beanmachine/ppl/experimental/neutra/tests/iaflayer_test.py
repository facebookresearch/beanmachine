# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
import torch.nn as nn
from beanmachine.ppl.experimental.neutra.iaflayer import InverseAutoregressiveLayer
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder


class IAFLayerTest(unittest.TestCase):
    def test_forward_backward_and_jacobian(self):

        hidden_layer = 4
        n_block = 1
        in_layer = 2
        out_layer = 2 * in_layer
        seed_num = 11

        rng = np.random.RandomState(0)
        x = torch.tensor(rng.rand(1, 2).astype(np.float32), requires_grad=True)
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        model = InverseAutoregressiveLayer(network_architecture, 0, True, 2)
        xtrhat, ja = model(x)
        mu, loga = model.get_parameter(x)

        z = x * torch.exp(loga) + (1 - torch.exp(loga)) * mu
        # calculate if x->z is correct.
        self.assertCountEqual(xtrhat.tolist(), z.tolist())
        # check if jacobian is correct.
        self.assertEqual(ja, loga.sum())

        # Here we don't expect to have a stable inverse direction,
        # so we do not need to check the inverse direction.
