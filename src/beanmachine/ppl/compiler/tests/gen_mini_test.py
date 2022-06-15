# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta, Normal


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip(n):
    return Bernoulli(beta() * 0.5)


@bm.random_variable
def normal(n):
    return Normal(flip(n), 1.0)


class CoinFlipTest(unittest.TestCase):
    def test_gen_mini(self) -> None:
        self.maxDiff = None
        # In the MiniBMG graph, the fact that we've observed
        # the flip(0) input to Normal(flip(0), 1.0) should ensure
        # that it is emitted into the graph as Normal(0.0, 1.0)
        queries = [beta(), normal(0), normal(1)]
        observations = {
            flip(0): tensor(0.0),
        }
        observed = BMGInference()._to_mini(queries, observations, indent=2)
        expected = """
{
  "comment": "Mini BMG",
  "nodes": [
    {
      "operator": "CONSTANT",
      "type": "REAL",
      "value": 2.0,
      "sequence": 0
    },
    {
      "operator": "DISTRIBUTION_BETA",
      "type": "DISTRIBUTION",
      "in_nodes": [
        0,
        0
      ],
      "sequence": 1
    },
    {
      "operator": "SAMPLE",
      "type": "REAL",
      "in_nodes": [
        1
      ],
      "sequence": 2
    },
    {
      "operator": "CONSTANT",
      "type": "REAL",
      "value": 0.5,
      "sequence": 3
    },
    {
      "operator": "MULTIPLY",
      "type": "REAL",
      "in_nodes": [
        2,
        3
      ],
      "sequence": 4
    },
    {
      "operator": "DISTRIBUTION_BERNOULLI",
      "type": "DISTRIBUTION",
      "in_nodes": [
        4
      ],
      "sequence": 5
    },
    {
      "operator": "CONSTANT",
      "type": "REAL",
      "value": 0.0,
      "sequence": 6
    },
    {
      "operator": "OBSERVE",
      "type": "NONE",
      "in_nodes": [
        5,
        6
      ],
      "sequence": 7
    },
    {
      "operator": "QUERY",
      "type": "NONE",
      "query_index": 0,
      "in_nodes": [
        2
      ],
      "sequence": 8
    },
    {
      "operator": "CONSTANT",
      "type": "REAL",
      "value": 1.0,
      "sequence": 9
    },
    {
      "operator": "DISTRIBUTION_NORMAL",
      "type": "DISTRIBUTION",
      "in_nodes": [
        6,
        9
      ],
      "sequence": 10
    },
    {
      "operator": "SAMPLE",
      "type": "REAL",
      "in_nodes": [
        10
      ],
      "sequence": 11
    },
    {
      "operator": "QUERY",
      "type": "NONE",
      "query_index": 1,
      "in_nodes": [
        11
      ],
      "sequence": 12
    },
    {
      "operator": "SAMPLE",
      "type": "REAL",
      "in_nodes": [
        5
      ],
      "sequence": 13
    },
    {
      "operator": "DISTRIBUTION_NORMAL",
      "type": "DISTRIBUTION",
      "in_nodes": [
        13,
        9
      ],
      "sequence": 14
    },
    {
      "operator": "SAMPLE",
      "type": "REAL",
      "in_nodes": [
        14
      ],
      "sequence": 15
    },
    {
      "operator": "QUERY",
      "type": "NONE",
      "query_index": 2,
      "in_nodes": [
        15
      ],
      "sequence": 16
    }
  ]
}
        """
        self.assertEqual(expected.strip(), observed.strip())
