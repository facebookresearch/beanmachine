# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for Graph from graph.py"""
import unittest

from beanmachine.ppl.utils.graph import Graph


class SimpleNode(object):
    name: str
    label: int

    def __init__(self, name: str, label: int):
        self.name = name
        self.label = label


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
        self.maxDiff = None
        #       a1    b1   c1
        #        |     |
        #       a2    b2
        #       / \   / \
        #     a5   s3    b5
        #          |
        #          s4
        #
        # a1 and b1 are isomorphic, a1 and c1 are not

        a1 = SimpleNode("a1", 1)
        b1 = SimpleNode("b1", 1)
        c1 = SimpleNode("c1", 1)
        a2 = SimpleNode("a2", 2)
        a5 = SimpleNode("a5", 5)
        b2 = SimpleNode("b2", 2)
        b5 = SimpleNode("b5", 5)
        s3 = SimpleNode("s3", 3)
        s4 = SimpleNode("s4", 4)
        g: Graph[SimpleNode] = Graph(
            lambda x: x.name, lambda x: str(x.label), lambda x: str(x.label)
        )
        g = g.with_edge(a1, a2).with_edge(a2, a5).with_edge(a2, s3)
        g = g.with_edge(b1, b2).with_edge(b2, s3).with_edge(b2, b5)
        g = g.with_edge(s3, s4)
        g = g.with_node(c1)

        self.assertTrue(g.are_dags_isomorphic(a1, b1))
        self.assertTrue(g.are_dags_isomorphic(a2, b2))
        self.assertFalse(g.are_dags_isomorphic(a1, c1))
        self.assertFalse(g.are_dags_isomorphic(a1, b2))

        reachable = ",".join(sorted(str(n.label) for n in g.reachable(b2)))
        self.assertEqual(reachable, "2,3,4,5")

        g.merge_isomorphic(a2, b2)
        # After merging b2 into a2:
        #     a1    b1   c1
        #       \  /
        #       a2
        #      / | \
        #    a5 s3  b5
        #        |
        #        s4

        observed = g.to_dot()
        expected = """
digraph "graph" {
  a1[label=1];
  a2[label=2];
  a5[label=5];
  b1[label=1];
  b5[label=5];
  c1[label=1];
  s3[label=3];
  s4[label=4];
  a1 -> a2;
  a2 -> a5;
  a2 -> b5;
  a2 -> s3;
  b1 -> a2;
  s3 -> s4;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_merge(self) -> None:
        self.maxDiff = None
        #           s1
        #         /  |  \
        #       a2   b2   c2
        #       / \  / \ /  \
        #     a3   a4  b3    b4
        #          |         |
        #          a5        b5
        #
        # The three "2" nodes are isomorphic.

        s1 = SimpleNode("s1", 1)
        a2 = SimpleNode("a2", 2)
        b2 = SimpleNode("b2", 2)
        c2 = SimpleNode("c2", 2)
        a3 = SimpleNode("a3", 3)
        a4 = SimpleNode("a4", 4)
        b3 = SimpleNode("b3", 3)
        b4 = SimpleNode("b4", 4)
        a5 = SimpleNode("a5", 5)
        b5 = SimpleNode("b5", 5)
        g: Graph[SimpleNode] = Graph(
            lambda x: x.name, lambda x: str(x.label), lambda x: str(x.label)
        )
        g = g.with_edge(s1, a2).with_edge(s1, b2).with_edge(s1, c2)
        g = g.with_edge(a2, a3).with_edge(a2, a4).with_edge(b2, a4)
        g = g.with_edge(b2, b3).with_edge(c2, b3).with_edge(c2, b4)
        g = g.with_edge(a4, a5).with_edge(b4, b5)

        observed = g.to_dot()
        expected = """
digraph "graph" {
  a2[label=2];
  a3[label=3];
  a4[label=4];
  a5[label=5];
  b2[label=2];
  b3[label=3];
  b4[label=4];
  b5[label=5];
  c2[label=2];
  s1[label=1];
  a2 -> a3;
  a2 -> a4;
  a4 -> a5;
  b2 -> a4;
  b2 -> b3;
  b4 -> b5;
  c2 -> b3;
  c2 -> b4;
  s1 -> a2;
  s1 -> b2;
  s1 -> c2;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

        g.merge_isomorphic_many([a2, b2, c2])

        observed = g.to_dot()
        #           s1
        #           |
        #           a2
        #       /  |   |   \
        #     a3   a4  b3  b4
        #          |       |
        #          a5      b5

        expected = """
digraph "graph" {
  a2[label=2];
  a3[label=3];
  a4[label=4];
  a5[label=5];
  b3[label=3];
  b4[label=4];
  b5[label=5];
  s1[label=1];
  a2 -> a3;
  a2 -> a4;
  a2 -> b3;
  a2 -> b4;
  a4 -> a5;
  b4 -> b5;
  s1 -> a2;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

        g.merge_isomorphic_children(a2)

        #           s1
        #           |
        #           a2
        #          /  \
        #         a3   a4
        #             /  \
        #            a5   b5
        # Note that the isomorphic 5 nodes are not recursively merged.

        observed = g.to_dot()
        expected = """
digraph "graph" {
  a2[label=2];
  a3[label=3];
  a4[label=4];
  a5[label=5];
  b5[label=5];
  s1[label=1];
  a2 -> a3;
  a2 -> a4;
  a4 -> a5;
  a4 -> b5;
  s1 -> a2;
}
        """
        self.assertEqual(observed.strip(), expected.strip())
