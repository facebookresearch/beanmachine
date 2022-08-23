# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference import BMGInference


@bm.random_variable
def foo():
    return dist.Normal(0.0, 1.0)


@bm.random_variable
def bat():
    return dist.Normal(0.0, 10.0)


@bm.random_variable
def bar(i):
    stmt: float = 1.2 * foo() + bat()
    return dist.Normal(stmt, 1.0)


class AnnotatedAssignmentTest(unittest.TestCase):
    def test_annotated_assignemnt(self) -> None:
        bat_value = dist.Normal(0.0, 10.0).sample(torch.Size((1, 1)))
        foo_value = dist.Normal(0.0, 1.0).sample(torch.Size((1, 1)))
        observations = {}
        bar_parent = dist.Normal(foo_value + bat_value, torch.tensor(1.0))
        for i in range(0, 1):
            observations[bar(i)] = bar_parent.sample(torch.Size((1, 1)))

        observed = BMGInference().to_dot(
            queries=[foo(), bat()],
            observations=observations,
        )
        print(observed)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=10.0];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1.2];
  N08[label="*"];
  N09[label="+"];
  N10[label=Normal];
  N11[label=Sample];
  N12[label="Observation 12.937742233276367"];
  N13[label=Query];
  N14[label=Query];
  N00 -> N02;
  N00 -> N05;
  N01 -> N02;
  N01 -> N10;
  N02 -> N03;
  N03 -> N08;
  N03 -> N13;
  N04 -> N05;
  N05 -> N06;
  N06 -> N09;
  N06 -> N14;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
