# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributions as dist
from beanmachine.ppl import random_variable
from beanmachine.ppl.utils.typeguards import is_rvidentifier_dict, is_rvidentifier_list


class TypeGuardsTests(unittest.TestCase):
    def setUp(self) -> None:
        @random_variable
        def alpha() -> dist.Distribution:
            return dist.Normal(0, 1)

        @random_variable
        def beta() -> dist.Distribution:
            return dist.Beta(1, 1)

        self.true_list = [alpha(), beta()]
        self.true_dict = {alpha(): torch.ones(2), beta(): torch.zeros(3)}

    def test_is_rvidentifier_list(self) -> None:
        self.assertTrue(is_rvidentifier_list(self.true_list))
        self.assertFalse(is_rvidentifier_list(self.true_list + [None]))
        self.assertFalse(is_rvidentifier_list(self.true_list + [torch.ones(2)]))
        self.assertFalse(is_rvidentifier_list([]))

    def test_is_rvidentifier_dict(self) -> None:
        self.assertTrue(is_rvidentifier_dict(self.true_dict))
        self.assertFalse(
            is_rvidentifier_dict({**self.true_dict, **{"a": torch.ones(5)}})
        )
        self.assertFalse(is_rvidentifier_dict({}))
