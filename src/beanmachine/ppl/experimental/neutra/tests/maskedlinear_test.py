# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from beanmachine.ppl.experimental.neutra.maskedlinear import MaskedLinear


class MaskedLinearGradTest(unittest.TestCase):
    def test_masked_value_and_grad(self):
        in_feature = 1
        out_feature = 6
        layer_ = MaskedLinear(in_feature, out_feature)
        mask_ = torch.ones(in_feature, out_feature)
        layer_.set_mask(mask_)

        # check if the mask shape is correct, and check if the masked initialize value is correect
        self.assertCountEqual(
            layer_.mask.t().tolist(), torch.ones(in_feature, out_feature).tolist()
        )

        # do a backpropagation to get masked_weight updated
        rng = np.random.RandomState(0)
        inp = torch.tensor(rng.rand(1, 6).astype(np.float32), requires_grad=True)
        loss = layer_(inp.t()).sum()
        loss.backward()

        truth = layer_.masked_weight
        previous_grad = layer_.weight.grad[:]

        # change one item in the mask to 0, and see if masked_weight can update properly.

        mask_[0][1] = 0
        layer_.set_mask(mask_)

        inp = torch.tensor(rng.rand(1, 6).astype(np.float32), requires_grad=True)
        loss = layer_(inp.t()).sum()
        loss.backward()

        truth[1][0] = 0
        updated_grad = layer_.weight.grad
        self.assertCountEqual(layer_.masked_weight.tolist(), truth.tolist())
        self.assertEqual(previous_grad[1], updated_grad[1])


class MaskedLinearShapeTest(unittest.TestCase):
    def test_mask_shape_negative(self):
        in_features = 1
        out_features = 6
        layer_ = MaskedLinear(in_features, out_features)
        mask_ = torch.ones(2, 6)
        with self.assertRaisesRegex(
            ValueError, "Dimension mismatches between mask and layer."
        ):
            layer_.set_mask(mask_)


class MaskedLinearNegativeTest(unittest.TestCase):
    def test_arg_must_vaild(self):
        in_features = 0
        out_features = 6
        with self.assertRaisesRegex(
            ValueError, "input feature dimension must be larger than 0"
        ):
            MaskedLinear(in_features, out_features)
