# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for Graph from graph.py"""
import unittest

from beanmachine.ppl.utils.graph import Graph


class X(object):
    x: int

    def __init__(self, x: int):
        self.x = x


class GraphTest(unittest.TestCase):
    def test_graph(self) -> None:
        self.maxDiff = None
        """Tests for Graph from graph.py"""
        g: Graph[str] = Graph(str, str)
        p1 = g.with_plate()
        p1.with_edge("a1", "a2").with_edge("a2", "a3")
        p2 = p1.with_plate()
        p2.with_edge("a0", "a1").with_edge("a3", "a0")
        p3 = g.with_plate()
        p3.with_edge("b0", "b1").with_edge("b1", "b2").with_edge("b2", "b3")
        p3.with_edge("b2", "a3").with_edge("a1", "b3")
        g.with_edge("start", "a0").with_edge("start", "b0")
        g.with_edge("a3", "end").with_edge("b3", "end")

        observed = g.to_dot()
        expected = """
digraph "graph" {
  a0;
  a1;
  a2;
  a3;
  b0;
  b1;
  b2;
  b3;
  end;
  start;
  a0 -> a1;
  a1 -> a2;
  a1 -> b3;
  a2 -> a3;
  a3 -> a0;
  a3 -> end;
  b0 -> b1;
  b1 -> b2;
  b2 -> a3;
  b2 -> b3;
  b3 -> end;
  start -> a0;
  start -> b0;
  subgraph cluster__0 {
    a1;
    a2;
    a3;
    subgraph cluster__0_0 {
      a0;
    }
  }
  subgraph cluster__1 {
    b0;
    b1;
    b2;
    b3;
  }
}
"""

        self.assertEqual(observed.strip(), expected.strip())

    def test_isomorphism(self) -> None:
        #       a1    b1   c1
        #        |     |
        #       a2    b2
        #       / \   / \
        #     a5   s3    b5
        #          |
        #          s4
        #
        # a1 and b1 are isomorphic, a1 and c1 are not

        a1 = X(1)
        b1 = X(1)
        c1 = X(1)
        a2 = X(2)
        a5 = X(5)
        b2 = X(2)
        b5 = X(5)
        s3 = X(3)
        s4 = X(4)
        g: Graph[X] = Graph(to_kernel=lambda x: str(x.x))
        g = g.with_edge(a1, a2).with_edge(a2, a5).with_edge(a2, s3)
        g = g.with_edge(b1, b2).with_edge(b2, s3).with_edge(b2, b5)
        g = g.with_edge(s3, s4)
        g = g.with_node(c1)

        self.assertTrue(g.are_dags_isomorphic(a1, b1))
        self.assertTrue(g.are_dags_isomorphic(a2, b2))
        self.assertFalse(g.are_dags_isomorphic(a1, c1))
        self.assertFalse(g.are_dags_isomorphic(a1, b2))

        reachable = ",".join(sorted(str(n.x) for n in g.reachable(b2)))
        self.assertEqual(reachable, "2,3,4,5")
