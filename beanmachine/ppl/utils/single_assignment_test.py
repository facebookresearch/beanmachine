# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for single_assignment.py"""
import ast
import unittest

import astor
from beanmachine.ppl.utils.ast_patterns import ast_domain
from beanmachine.ppl.utils.fold_constants import fold
from beanmachine.ppl.utils.rules import FirstMatch as first, TryMany as many
from beanmachine.ppl.utils.single_assignment import SingleAssignment, single_assignment


_some_top_down = ast_domain.some_top_down


class SingleAssignmentTest(unittest.TestCase):
    def test_single_assignment_0(self) -> None:
        """Really basic tests for single_assignment.py"""

        # Sanity check. Check that  if you change one of the two numbers the test fails
        self.assertEqual(3, 3)

        s = SingleAssignment()
        root = "root"
        name = s._unique_id(root)
        self.assertEqual(root, name[0 : len(root)])

        # Check to make sure that we need a new rule to handle unassigned expressions
        s = SingleAssignment()
        s._rules = many(  # Custom wire rewrites to rewrites existing before this diff
            _some_top_down(
                first([s._handle_return(), s._handle_for(), s._handle_assign()])
            )
        )
        source = """
def f(x):
    g(x)+x
"""
        expected = """
def f(x):
    g(x) + x
"""
        m = ast.parse(source)
        result = s.single_assignment(fold(m))
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        # Check that the unassigned expressions rule (unExp) works alone
        s = SingleAssignment()
        s.count = 0
        s._rules = many(_some_top_down(first([s._handle_unassigned()])))
        source = """
def f(x):
    g(x)+x
"""
        expected = """
def f(x):
    u1 = g(x) + x
"""
        m = ast.parse(source)
        result = s.single_assignment(fold(m))
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        # Check that the unassigned expressions rule (unExp) works in context
        s = SingleAssignment()
        s.count = 0
        s._rules = many(  # Custom wire some rewrites that include unExp
            _some_top_down(
                first(
                    [
                        s._handle_unassigned(),
                        s._handle_return(),
                        s._handle_for(),
                        s._handle_assign(),
                    ]
                )
            )
        )
        source = """
def f(x):
    g(x)+x
"""
        expected = """
def f(x):
    a2 = g(x)
    u1 = a2 + x
"""
        m = ast.parse(source)
        result = s.single_assignment(fold(m))
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_single_assignment(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = """
def f():
    if a and b:
        return 1 + ~x + 2 + g(5, y=6)
    z = torch.tensor([1.0 + 2.0, 4.0])
    for x in [[10, 20], [30, 40]]:
        for y in x:
            print(x + y)
    return 8 * y / (4 * z)
"""
        m = ast.parse(source)
        result = single_assignment(fold(m))
        expected = """
def f():
    if a and b:
        a9 = 3
        a15 = ~x
        a5 = a9 + a15
        a16 = 5
        a20 = 6
        a10 = g(a16, y=a20)
        r1 = a5 + a10
        return r1
    a2 = torch.tensor
    a11 = 3.0
    a17 = 4.0
    a6 = [a11, a17]
    z = a2(a6)
    a12 = 10
    a18 = 20
    a7 = [a12, a18]
    a19 = 30
    a21 = 40
    a13 = [a19, a21]
    f3 = [a7, a13]
    for x in f3:
        for y in x:
            print(x + y)
    a14 = 2.0
    a8 = a14 * y
    r4 = a8 / z
    return r4
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_single_assignment_2(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = "b = c(d + e).f(g + h)"
        m = ast.parse(source)
        result = single_assignment(m)
        expected = """
a4 = d + e
a2 = c(a4)
a1 = a2.f
a3 = g + h
b = a1(a3)
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_single_assignment_3(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = "a = (b+c)[f(d+e)]"
        m = ast.parse(source)
        result = single_assignment(m)
        expected = """
a1 = b + c
a3 = d + e
a2 = f(a3)
a = a1[a2]
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())
