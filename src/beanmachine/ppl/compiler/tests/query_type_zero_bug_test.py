# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def prod_1(counter):
    prod = 0.0
    for i in range(counter):
        prod = prod * norm(i)
    return prod


@bm.functional
def prod_2():
    return prod_1(10)


class ZeroQueryTypeCheckingBug(unittest.TestCase):
    def test_query_type_zero(self) -> None:
        """
        Query of a variable of type Zero produces a type checking error.
        """
        self.maxDiff = None

        # TODO: One of the design principles of BMG is to allow
        # TODO: for any query, even if you ask it to query constants.
        # TODO: A potential solution could be to add a warning system so that
        # TODO: the model's developer becomes aware of the possible error
        with self.assertRaises(AssertionError) as ex:
            BMGInference().infer([prod_2()], {}, 1)
        expected = ""
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())
