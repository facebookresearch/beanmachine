# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for print_graph from dotbuilder.py"""
import unittest
from typing import Any, Dict

from beanmachine.ppl.utils.dotbuilder import DotBuilder, print_graph


class GraphPrinterTest(unittest.TestCase):
    def test_print_tree(self) -> None:
        """Tests for print_graph from dotbuilder.py"""
        bar = {"blah": [2, 3, {"abc": (6, 7, (5, 5, 6))}]}
        d: Dict[Any, Any] = {"foo": 2, "bar1": bar, "bar2": bar}
        d["self"] = d  # type: ignore
        observed = print_graph([d])
        expected = """
digraph "graph" {
  N0[label=dict];
  N10[label=5];
  N1[label=2];
  N2[label=dict];
  N3[label=list];
  N4[label=3];
  N5[label=dict];
  N6[label=tuple];
  N7[label=6];
  N8[label=7];
  N9[label=tuple];
  N0 -> N0[label=self];
  N0 -> N1[label=foo];
  N0 -> N2[label=bar1];
  N0 -> N2[label=bar2];
  N2 -> N3[label=blah];
  N3 -> N1[label=0];
  N3 -> N4[label=1];
  N3 -> N5[label=2];
  N5 -> N6[label=abc];
  N6 -> N7[label=0];
  N6 -> N8[label=1];
  N6 -> N9[label=2];
  N9 -> N10[label=0];
  N9 -> N10[label=1];
  N9 -> N7[label=2];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_builder(self) -> None:
        self.maxDiff = None
        db = DotBuilder("my_graph")
        db.with_comment("comment")

        db.start_subgraph("my_subgraph", True)
        db.with_label("graph_label")
        db.with_node("A1", "A")
        db.with_node("A2", "A")
        db.with_edge("A1", "A2", "edge_label")
        db.end_subgraph()
        observed = str(db)
        expected = """
digraph my_graph {
  // comment
  subgraph cluster_my_subgraph {
    label=graph_label
    A1[label=A];
    A2[label=A];
    A1 -> A2[label=edge_label];
  }
}
        """
        self.assertEqual(observed.strip(), expected.strip())
