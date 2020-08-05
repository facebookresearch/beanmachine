# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for ast_tools.py"""
import ast
import unittest

import beanmachine.ppl.compiler.ast_tools as ast_tools


class ASTToolsTest(unittest.TestCase):
    def test_ast_tools_print_tree(self) -> None:
        """test_ast_tools_print_tree"""
        node = ast.parse("2 + 3")
        observed = ast_tools.print_tree(node, False)
        expected = """
Module
+-list
  +-Expr
    +-BinOp
      +-Num
      | +-2
      +-Add
      +-Num
        +-3
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

    def test_ast_tools_print_graph(self) -> None:
        """test_ast_tools_print_graph"""
        node = ast.parse("2 + 3")
        observed = ast_tools.print_graph(node)
        expected = """
digraph "graph" {
  N0[label=Module];
  N1[label=list];
  N2[label=Expr];
  N3[label=BinOp];
  N4[label=Num];
  N5[label=Add];
  N6[label=Num];
  N7[label=3];
  N8[label=2];
  N0 -> N1[label=body];
  N1 -> N2[label=0];
  N2 -> N3[label=value];
  N3 -> N4[label=left];
  N3 -> N5[label=op];
  N3 -> N6[label=right];
  N4 -> N8[label=n];
  N6 -> N7[label=n];
}"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

    def test_ast_tools_print_python(self) -> None:
        """test_ast_tools_print_python"""
        node = ast.parse("x = f(2 + 3)")
        observed = ast_tools.print_python(node)
        expected = """
Module(
    body=[
        Assign(
            targets=[Name(id="x", ctx=Store())],
            value=Call(
                func=Name(id="f", ctx=Load()),
                args=[BinOp(left=Num(n=2), op=Add(), right=Num(n=3))],
                keywords=[],
            ),
        )
    ]
)
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())
