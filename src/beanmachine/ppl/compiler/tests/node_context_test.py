# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch.distributions import Bernoulli, Beta


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip1(n):
    return Bernoulli(beta())


@bm.functional
def sum1():
    return flip1(0) + 1.0


def sum2(n, m):
    # Note that sum2 is NOT a functional.
    # The returned addition node should deduplicate with
    # the one returned by sum1().
    return flip1(0) + (n * m)


@bm.functional
def prod1(n):
    # Try a named argument.
    return sum1() * sum2(1.0, m=1.0)


@bm.functional
def log1(n):
    return prod1(n).log()


def _dict_to_str(d) -> str:
    return "\n".join(
        sorted(
            type(key).__name__ + ":{" + ",".join(sorted(str(v) for v in d[key])) + "}"
            for key in d
        )
    )


class NodeContextTest(unittest.TestCase):
    def test_node_context(self) -> None:
        self.maxDiff = None
        rt = BMGRuntime()
        rt.accumulate_graph([log1(123)], {})
        expected = """
AdditionNode:{sum1(),sum2(1.0,m=1.0)}
BernoulliNode:{flip1(0)}
BetaNode:{beta()}
LogNode:{log1(123)}
MultiplicationNode:{prod1(123)}
SampleNode:{beta()}
SampleNode:{flip1(0)}
"""
        observed = _dict_to_str(rt._context._node_locations)
        self.assertEqual(expected.strip(), observed.strip())
