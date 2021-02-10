# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for single_assignment.py"""
import ast
import unittest

import astor
from beanmachine.ppl.utils.ast_patterns import ast_domain
from beanmachine.ppl.utils.fold_constants import fold
from beanmachine.ppl.utils.rules import (
    FirstMatch as first,
    TryMany as many,
    TryOnce as once,
)
from beanmachine.ppl.utils.single_assignment import SingleAssignment


_some_top_down = ast_domain.some_top_down


class SingleAssignmentTest(unittest.TestCase):

    s = SingleAssignment()
    default_rule = s._rule
    default_rules = s._rules

    def test_single_assignment_sanity_check(self) -> None:
        """If you manually change one of the two numbers in the test it should fail"""

        self.assertEqual(3, 3)

    def test_single_assignment_unique_id_preserves_prefix(self) -> None:
        """The method unique_id preserves name prefix"""

        s = SingleAssignment()
        root = "root"
        name = s._unique_id(root)
        self.assertEqual(root, name[0 : len(root)])

    def check_rewrite(self, source, expected, rules=default_rules, msg=None):
        """Applying rules to source yields expected"""

        self.maxDiff = None
        self.s._count = 0
        self.s._rules = rules
        m = ast.parse(source)
        result = self.s.single_assignment(fold(m))
        self.assertEqual(astor.to_source(result).strip(), expected.strip(), msg=msg)

    def check_rewrites(self, sources, rule=default_rule):
        """Applying rules to each element of sources yelds the next one"""

        self.assertIsInstance(sources, list, msg="\nSources should be list of strings.")
        self.assertGreater(len(sources), 0, msg="\nSources should be a non-empty list.")
        if len(sources) == 1:
            return self.check_rewrite(
                sources[0],
                sources[0],
                once(_some_top_down(rule)),
                msg="\nExpected the term to be a normal form for rule.",
            )
        source, *rest = sources
        expected, *_ = rest
        self.check_rewrite(
            source,
            expected,
            _some_top_down(rule),
            msg="\nExpected rule to rewrite one term to the other",
        )
        self.check_rewrites(rest, rule)

    def test_check_rewrites(self) -> None:
        """The method check_rewrites performs several rewrites for it in one shot.
        This method illustrates these functions."""

        # The tests use a running example consisting of three terms that are the first,
        # intermediate, and final terms in a sequence of rewrites by the rule
        # self.s._handle_boolop_binarize()
        # The three terms are simple as follows:

        source1 = """
def f(x):
    x = a and b and c and d
"""
        source2 = """
def f(x):
    x = (a and b) and c and d
"""
        source3 = """
def f(x):
    x = ((a and b) and c) and d
"""
        # First, check that it raises errors on bad inputs
        with self.assertRaises(
            AssertionError, msg="The following line should raise an error!"
        ):
            self.check_rewrites(42, self.s._handle_boolop_binarize())

        with self.assertRaises(
            AssertionError, msg="The following line should raise an error!"
        ):
            self.check_rewrites([], self.s._handle_boolop_binarize())

        # Second, make sure it it does what is expected on normal forms
        self.check_rewrites([source3], self.s._handle_boolop_binarize())

        with self.assertRaises(
            AssertionError, msg="The following line should raise an error!"
        ):
            self.check_rewrites([source1], self.s._handle_boolop_binarize())

        # Third, normal forms are unchanged if we have one "many" too many
        self.check_rewrites([source3], many(self.s._handle_boolop_binarize()))

        with self.assertRaises(
            AssertionError, msg="The following line should raise an error!"
        ):
            self.check_rewrites([source1], many(self.s._handle_boolop_binarize()))

        # Fourth, it will recognize valid rewrites
        self.check_rewrites(
            [source1, source2, source3], self.s._handle_boolop_binarize()
        )

        # In common use, it is expect that the intermediate expressions are
        # all gathered in a list (if we would like to test the sequence in)
        # multiple ways, or they may be inlined directly. To get a sense of
        # the way the automatic formatting renders such uses, we include both
        # here:

        sources = [
            """
def f(x):
    x = a and b and c and d
""",
            """
def f(x):
    x = (a and b) and c and d
""",
            """
def f(x):
    x = ((a and b) and c) and d
""",
        ]

        self.check_rewrites(sources, self.s._handle_boolop_binarize())

        # and

        self.check_rewrites(
            [
                """
def f(x):
    x = a and b and c and d
""",
                """
def f(x):
    x = (a and b) and c and d
""",
                """
def f(x):
    x = ((a and b) and c) and d
""",
            ],
            self.s._handle_boolop_binarize(),
        )

        # Both forms are a bit verbose, but the first is somewhat more passable

        # Fifth, the above call is essentially the following reduction but
        # with the intermediate term(s) spelled out:
        self.check_rewrite(
            source1, source3, many(_some_top_down(self.s._handle_boolop_binarize()))
        )

        # Sixth, we can use the default rules to document full reduction sequences

        sources_continued = [
            source3,
            """
def f(x):
    a1 = (a and b) and c
    x = a1 and d
""",
            """
def f(x):
    a1 = a and b
    a1 = a1 and c
    if a1:
        x = d
    else:
        x = a1
""",
            """
def f(x):
    if a:
        a1 = b
    else:
        a1 = a
    if a1:
        a1 = c
    else:
        a1 = a1
    if a1:
        x = d
    else:
        x = a1
""",
        ]

        self.check_rewrites(sources_continued)

        # TODO: Remarks based on the sequence above:
        # 1) At some point we may decide to use top_down rather than some_top_down
        # 2) There is a potential new nameshadowing problem

    def check_rewrite_as_ast(self, source, expected, rules=default_rules):
        """Applying rules to source yields expected -- checked as ASTs"""

        self.maxDiff = None
        self.s._count = 0
        m = ast.parse(source)
        result = self.s.single_assignment(fold(m))
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))

    def test_single_assignment_pre_unassigned_expressions(self) -> None:
        """Tests for state before adding rule to handle unassigned expressions"""

        source = """
def f(x):
    g(x)+x
"""
        expected = """
def f(x):
    g(x) + x
"""

        self.check_rewrite(
            source,
            expected,
            many(  # Custom wire rewrites to rewrites existing before this diff
                _some_top_down(
                    first(
                        [
                            self.s._handle_return(),
                            self.s._handle_for(),
                            self.s._handle_assign(),
                        ]
                    )
                )
            ),
        )

    def test_single_assignment_unassigned_expressions(self) -> None:
        """Test unassiged expressions rewrite"""

        # Check that the unassigned expressions rule (unExp) works alone

        source = """
def f(x):
    g(x)+x
"""
        expected = """
def f(x):
    u1 = g(x) + x
"""
        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_unassigned())
        )

        # Check that the unassigned expressions rule (unExp) works in context

        source = """
def f(x):
    g(x)+x
"""
        expected = """
def f(x):
    r3 = [x]
    r4 = {}
    a2 = g(*r3, **r4)
    u1 = a2 + x
"""

        self.check_rewrite(
            source,
            expected,
            many(
                _some_top_down(
                    first(
                        [
                            self.s._handle_unassigned(),
                            self.s._handle_return(),
                            self.s._handle_for(),
                            self.s._handle_assign(),
                        ]
                    )
                )
            ),
        )

    def test_single_assignment_if(self) -> None:
        """Test if rewrite"""

        # Check that rule will leave uninteresting expressions alone

        source = """
def f(x):
    if x:
        c=a+b+c
    else:
        b=c+a+b
"""
        expected = """
def f(x):
    if x:
        c = a + b + c
    else:
        b = c + a + b
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(first([self.s._handle_if()])))
        )

        # Check that the if rule works (alone) on an elementary expression

        source = """
def f(x):
    if x+x>x:
        c=a+b+c
    else:
        b=c+a+b
"""
        expected = """
def f(x):
    r1 = x + x > x
    if r1:
        c = a + b + c
    else:
        b = c + a + b
"""
        self.check_rewrite(source, expected, _some_top_down(self.s._handle_if()))

        # Check that the if rule works (alone) with elif clauses

        source = """
def f(x):
    if x+x>x:
        c=a+b+c
    elif y+y>y:
        a=c+b+a
    else:
        b=c+a+b
"""
        expected = """
def f(x):
    r1 = x + x > x
    if r1:
        c = a + b + c
    else:
        r2 = y + y > y
        if r2:
            a = c + b + a
        else:
            b = c + a + b
"""
        self.check_rewrite(source, expected, many(_some_top_down(self.s._handle_if())))

        # Check that the if rule works (with others) on an elementary expression

        source = """
def f(x):
    if gt(x+x,x):
        c=a+b+c
    else:
        b=c+a+b
"""
        expected = """
def f(x):
    a6 = x + x
    a5 = [a6]
    a7 = [x]
    r4 = a5 + a7
    r8 = {}
    r1 = gt(*r4, **r8)
    if r1:
        a2 = a + b
        c = a2 + c
    else:
        a3 = c + a
        b = a3 + b
"""
        self.check_rewrite(
            source,
            expected,
            many(
                _some_top_down(
                    first(
                        [
                            self.s._handle_if(),
                            self.s._handle_unassigned(),
                            self.s._handle_return(),
                            self.s._handle_for(),
                            self.s._handle_assign(),
                        ]
                    )
                )
            ),
        )

    def test_single_assignment_while(self) -> None:
        """Test while rewrite"""

        # Check that while_not_True rule works (alone) on simple cases

        source = """
def f(x):
    while c:
        x=x+1
"""
        expected = """
def f(x):
    while True:
        w1 = c
        if w1:
            x = x + 1
        else:
            break
"""
        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_while_not_True())
        )

        # Check that the while_not_True rewrite reaches normal form
        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_while_not_True()))
        )

        # Check that while_not_True_else rule works (alone) on simple cases

        source = """
def f(x):
    while c:
        x=x+1
    else:
        x=x-1
"""
        expected = """
def f(x):
    while True:
        w1 = c
        if w1:
            x = x + 1
        else:
            break
    if not w1:
        x = x - 1
"""
        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_while_not_True_else())
        )

        # Check that the while_not_True_else rewrite reaches normal form
        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_while_not_True_else()))
        )

        # Check that while_True_else rule works (alone) on simple cases

        source = """
def f(x):
    while True:
        x=x+1
    else:
        x=x-1
"""
        expected = """
def f(x):
    while True:
        x = x + 1
"""
        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_while_True_else())
        )

        # Check that while_True_else rule, alone, on simple cases, reaches a normal form

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_while_True_else()))
        )

        # Check that (combined) while rule works (alone) on simple cases

        source = """
def f(x):
    while c:
        x=x+1
    while d:
        y=y+1
    else:
        y=y-1
    while True:
        z=z+1
    else:
        z=z-1

"""
        expected = """
def f(x):
    while True:
        w1 = c
        if w1:
            x = x + 1
        else:
            break
    while True:
        w2 = d
        if w2:
            y = y + 1
        else:
            break
    if not w2:
        y = y - 1
    while True:
        z = z + 1
"""

        self.check_rewrite(source, expected, _some_top_down(self.s._handle_while()))

        # Extra check: Make sure they are idempotent
        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_while()))
        )

        # Check that the while rewrite works with everything else

        self.maxDiff = None

        source = """
def f(x):
    while c:
        x=(x+1)-s
    else:
        x=(x-1)+s
    while True:
        y=(y+1)-s
    else:
        y=(y-1)+s

"""
        expected = """
def f(x):
    while True:
        w1 = c
        if w1:
            a5 = 1
            a2 = x + a5
            x = a2 - s
        else:
            break
    r3 = not w1
    if r3:
        a8 = 1
        a6 = x - a8
        x = a6 + s
    while True:
        a7 = 1
        a4 = y + a7
        y = a4 - s
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_boolop_binarize(self) -> None:
        """Test the rule for converting boolean operators into binary operators"""

        source = """
def f(x):
    x = a and b and c and d
"""
        expected = """
def f(x):
    x = ((a and b) and c) and d
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_boolop_binarize()))
        )

        source = """
def f(x):
    x = a and b and c or d or e
"""
        expected = """
def f(x):
    x = ((a and b) and c or d) or e
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_boolop_binarize()))
        )

    def test_single_assignment_boolop_linearize(self) -> None:
        """Test the assign rule for linearizing binary boolean ops"""

        source = """
def f(x):
    x = (a and b) and c
"""
        expected = """
def f(x):
    a1 = a and b
    x = a1 and c
"""

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_boolop_linearize())),
        )

        source = """
def f(x):
    x = ((a and b) and c) and d
"""
        expected = """
def f(x):
    a2 = a and b
    a1 = a2 and c
    x = a1 and d
"""

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_boolop_linearize())),
        )

    def test_single_assignment_and2if(self) -> None:
        """Test the assign rule for converting a binary and into an if statement"""

        source = """
def f(x):
    x = a and b
"""
        expected = """
def f(x):
    if a:
        x = b
    else:
        x = a
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_and2if()))
        )

    def test_single_assignment_or2if(self) -> None:
        """Test the assign rule for converting a binary or into an if statement"""

        source = """
def f(x):
    x = a or b
"""
        expected = """
def f(x):
    if a:
        x = a
    else:
        x = b
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_or2if()))
        )

    def test_single_assignment_boolop_all(self) -> None:
        """Test the combined rules for boolean operators"""

        source = """
def f(x):
    x = a and b and c and d
"""
        expected = """
def f(x):
    if a:
        a2 = b
    else:
        a2 = a
    if a2:
        a1 = c
    else:
        a1 = a2
    if a1:
        x = d
    else:
        x = a1
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_boolop_all()))
        )

        source = """
def f(x):
    x = a and b and c or d or e
"""
        expected = """

def f(x):
    if a:
        a3 = b
    else:
        a3 = a
    if a3:
        a2 = c
    else:
        a2 = a3
    if a2:
        a1 = a2
    else:
        a1 = d
    if a1:
        x = a1
    else:
        x = e
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_boolop_all()))
        )

    def test_single_assignment_handle_compare_binarize(self) -> None:
        """Test the rule for converting n-way comparisons into binary ones"""

        source = """
def f(x):
    x = a < b > c == d
"""
        expected = """
def f(x):
    x = a < b and (b > c and c == d)
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_compare_binarize()))
        )

        source = """
def f(x):
    x = a < 1 + b > c == d
"""
        expected = """
def f(x):
    x = a < 1 + b > c == d
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_compare_binarize()))
        )

        source = """
def f(x):
    x = a + 1 < b > c + 1 == d
"""
        expected = """
def f(x):
    x = a + 1 < b and b > c + 1 == d
"""  # Note that the term after the "and" is not reduced

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_compare_binarize()))
        )

    def test_single_assignment_handle_assign_compare_lefthandside(self) -> None:
        """Test the rule for lifting first argument of n-way comparison"""

        source = """
def f(x):
    x = 1 + a < 1 + b > c == d
"""
        expected = """
def f(x):
    a1 = 1 + a
    x = a1 < 1 + b > c == d
"""

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_compare_lefthandside())),
        )

    def test_single_assignment_handle_assign_compare_righthandside(self) -> None:
        """Test the rule for lifting second argument of n-way comparison"""

        source = """
def f(x):
    z = 1 + a
    x = z < 1 + b > c == d
"""
        expected = """
def f(x):
    z = 1 + a
    a1 = 1 + b
    x = z < a1 > c == d
"""

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_compare_righthandside())),
        )

    def test_single_assignment_handle_assign_compare_bothhandsides(self) -> None:
        """Test the rules for lifting first and second args of n-way comparison"""

        source = """
def f(x):
    x = 1 + a < 1 + b > c == d
"""
        expected = """
def f(x):
    a1 = 1 + a
    a2 = 1 + b
    x = a1 < a2 > c == d
"""

        self.check_rewrite(
            source,
            expected,
            many(
                _some_top_down(
                    first(
                        [
                            self.s._handle_assign_compare_lefthandside(),
                            self.s._handle_assign_compare_righthandside(),
                        ]
                    )
                )
            ),
        )

    def test_single_assignment_handle_assign_compare_all(self) -> None:
        """Test alls rules for n-way comparisons"""

        source = """
def f(x):
    x = 1 + a < 1 + b > c == d
"""
        expected = """
def f(x):
    a1 = 1 + a
    a2 = 1 + b
    x = a1 < a2 and (a2 > c and c == d)
"""

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_compare_all()))
        )

    def test_single_assignment_handle_assign_compare_all_combined(self) -> None:
        """Test alls rules for n-way comparisons combined with rest"""

        source = """
def f(x):
    x = 1 + a < 1 + b > c == d
"""
        expected = """
def f(x):
    a2 = 1
    a1 = a2 + a
    a4 = 1
    a3 = a4 + b
    a5 = a1 < a3
    if a5:
        a6 = a3 > c
        if a6:
            x = c == d
        else:
            x = a6
    else:
        x = a5
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_lists(self) -> None:
        """Test the assign rule for lists"""

        source = """
def f(x):
    x = [1+a,a+b,c+d]
"""
        expected = """
def f(x):
    a2 = 1
    a1 = a2 + a
    a3 = a + b
    a4 = c + d
    x = [a1, a3, a4]
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_dict(self) -> None:
        """Test the assign rule for dictionaries"""

        source = """
def f(x):
    x = {"a"+"b":x+x}
"""
        expected = """
def f(x):
    a2 = 'a'
    a4 = 'b'
    a1 = a2 + a4
    a3 = x + x
    x = {a1: a3}
"""

        self.check_rewrite(source, expected)

        source = """
def f(x):
    x = {"a"+"b":x+x, "c"+"d":x-x}
"""
        expected = """
def f(x):
    a2 = 'a'
    a4 = 'b'
    a1 = a2 + a4
    a3 = x + x
    a6 = 'c'
    a8 = 'd'
    a5 = a6 + a8
    a7 = x - x
    x = {a1: a3, a5: a7}"""

        self.check_rewrite(source, expected)

    def test_single_assignment_tuple(self) -> None:
        """Test the assign rule for tuples"""

        source = """
def f(x):
    x = 1+a,a+b,c+d
"""
        expected = """
def f(x):
    a2 = 1
    a1 = a2 + a
    a3 = a + b
    a4 = c + d
    x = a1, a3, a4
"""

        self.check_rewrite(source, expected)

    def test_single_assignment(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = """
def f():
    aab = a + b
    if aab:
        return 1 + ~x + 2 + g(5, y=6)
    z = torch.tensor([1.0 + 2.0, 4.0])
    for x in [[10, 20], [30, 40]]:
        for y in x:
            _1 = x+y
            _2 = print(_1)
    return 8 * y / (4 * z)
"""

        expected = """
def f():
    aab = a + b
    if aab:
        a8 = 3
        a15 = ~x
        a5 = a8 + a15
        a24 = 5
        r19 = [a24]
        a27 = 6
        r26 = dict(y=a27)
        a9 = g(*r19, **r26)
        r1 = a5 + a9
        return r1
    a2 = torch.tensor
    a20 = 3.0
    a25 = 4.0
    a16 = [a20, a25]
    r10 = [a16]
    r21 = {}
    z = a2(*r10, **r21)
    a11 = 10
    a17 = 20
    a6 = [a11, a17]
    a18 = 30
    a22 = 40
    a12 = [a18, a22]
    f3 = [a6, a12]
    for x in f3:
        for y in x:
            _1 = x + y
            r13 = [_1]
            r23 = {}
            _2 = print(*r13, **r23)
    a14 = 2.0
    a7 = a14 * y
    r4 = a7 / z
    return r4
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_2(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = "b = c(d + e).f(g + h)"

        expected = """
a6 = d + e
r4 = [a6]
r8 = {}
a2 = c(*r4, **r8)
a1 = a2.f
a5 = g + h
r3 = [a5]
r7 = {}
b = a1(*r3, **r7)
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_3(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = "a = (b+c)[f(d+e)]"

        expected = """
a1 = b + c
a4 = d + e
r3 = [a4]
r5 = {}
a2 = f(*r3, **r5)
a = a1[a2]
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_call_single_star_arg(self) -> None:
        """Test the assign rule final step in rewriting regular call arguments"""

        source = """
x = f(*([1]+[2]))
"""
        expected = """
r1 = [1] + [2]
x = f(*r1)
"""

        self.check_rewrite(
            source,
            expected,
            _some_top_down(self.s._handle_assign_call_single_star_arg()),
        )

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_single_star_arg())),
        )

        expected = """
a3 = 1
a2 = [a3]
a6 = 2
a4 = [a6]
r1 = a2 + a4
r5 = {}
x = f(*r1, **r5)
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_call_single_double_star_arg(self) -> None:
        """Test the assign rule final step in rewriting keyword call arguments"""

        source = """
x = f(*d, **({x: 5}))
"""
        expected = """
r1 = {x: 5}
x = f(*d, **r1)
"""

        self.check_rewrite(
            source,
            expected,
            _some_top_down(self.s._handle_assign_call_single_double_star_arg()),
        )

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_single_double_star_arg())),
        )

        expected = """
a2 = 5
r1 = {x: a2}
x = f(*d, **r1)"""
        self.check_rewrite(source, expected)

    def test_single_assignment_call_two_star_args(self) -> None:
        """Test the assign rule for merging starred call arguments"""

        source = """
x = f(*[1],*[2])
"""
        expected = """
x = f(*([1] + [2]))
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_call_two_star_args())
        )

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_two_star_args())),
        )

        expected = """
a3 = 1
a2 = [a3]
a6 = 2
a4 = [a6]
r1 = a2 + a4
r5 = {}
x = f(*r1, **r5)
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_call_two_double_star_args(self) -> None:
        """Test the assign rule for merging double starred call arguments"""

        source = """
x = f(*d,**a, **b, **c)
"""
        expected = """
x = f(*d, **dict(**a, **b), **c)
"""

        self.check_rewrite(
            source,
            expected,
            _some_top_down(self.s._handle_assign_call_two_double_star_args()),
        )

        expected = """
x = f(*d, **dict(**dict(**a, **b), **c))
"""

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_two_double_star_args())),
        )

        source = expected
        expected = """
r1 = dict(**dict(**a, **b), **c)
x = f(*d, **r1)
"""

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_single_double_star_arg())),
        )

        expected = """
a2 = dict(**a, **b)
r1 = dict(**a2, **c)
x = f(*d, **r1)
"""
        self.check_rewrite(source, expected)

        source = """
x= f(**{a:1},**{b:3})
"""

        expected = """
x = f(**dict(**{a: 1}, **{b: 3}))
"""

        self.check_rewrite(
            source,
            expected,
            _some_top_down(self.s._handle_assign_call_two_double_star_args()),
        )

    def test_single_assignment_call_regular_arg(self) -> None:
        """Test the assign rule for starring an unstarred regular arg"""

        source = """
x = f(*[1], 2)
"""
        expected = """
x = f(*[1], *[2])
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_call_regular_arg())
        )

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_regular_arg())),
        )

        expected = """
a3 = 1
a2 = [a3]
a6 = 2
a4 = [a6]
r1 = a2 + a4
r5 = {}
x = f(*r1, **r5)
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_call_keyword_arg(self) -> None:
        """Test the assign rule for starring an unstarred keyword arg"""

        source = """
x = f(**dict(**d), k=42, **dict(**e))
"""
        expected = """
x = f(**dict(**d), **dict(k=42), **dict(**e))
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_call_keyword_arg())
        )

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_keyword_arg())),
        )

        # TODO: This just for debugging a non-terminating loop
        expected = """
x = f(*[], **dict(**d), k=42, **dict(**e))
"""
        self.check_rewrite(source, expected, _some_top_down(self.s._handle_assign()))

        source = expected
        expected = """
r1 = []
x = f(*r1, **dict(**d), k=42, **dict(**e))
"""
        self.check_rewrite(source, expected, _some_top_down(self.s._handle_assign()))

        source = expected
        expected = """
r1 = []
x = f(*r1, **dict(**d), **dict(k=42), **dict(**e))
"""
        self.check_rewrite(source, expected, _some_top_down(self.s._handle_assign()))

        source = expected
        expected = """
r1 = []
x = f(*r1, **dict(**dict(**d), **dict(k=42)), **dict(**e))
"""
        self.check_rewrite(source, expected, _some_top_down(self.s._handle_assign()))

        source = expected
        expected = """
r1 = []
x = f(*r1, **dict(**dict(**dict(**d), **dict(k=42)), **dict(**e)))
"""
        self.check_rewrite(source, expected, _some_top_down(self.s._handle_assign()))

        source = expected
        expected = """
r1 = []
a3 = dict(**d)
a6 = 42
a5 = dict(k=a6)
a2 = dict(**a3, **a5)
a4 = dict(**e)
r1 = dict(**a2, **a4)
x = f(*r1, **r1)
"""
        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign()))
        )

        source = """
x = f(**dict(**d), k=42, **dict(**e))
"""
        expected = """
r1 = []
a4 = dict(**d)
a7 = 42
a6 = dict(k=a7)
a3 = dict(**a4, **a6)
a5 = dict(**e)
r2 = dict(**a3, **a5)
x = f(*r1, **r2)
"""
        self.check_rewrite(source, expected)

        source = """
x = f()
"""
        expected = """
r1 = []
r2 = {}
x = f(*r1, **r2)
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_call_empty_regular_arg(self) -> None:
        """Test the assign rule for starring an empty regular arg"""

        source = """
x = f()
"""
        expected = """
x = f(*[])
"""

        self.check_rewrite(
            source,
            expected,
            _some_top_down(self.s._handle_assign_call_empty_regular_arg()),
        )

        self.check_rewrite(
            source,
            expected,
            many(_some_top_down(self.s._handle_assign_call_empty_regular_arg())),
        )

        expected = """
r1 = []
r2 = {}
x = f(*r1, **r2)
"""
        self.check_rewrite(source, expected)

    def test_single_assignment_call_three_arg(self) -> None:
        """Test the assign rule for starring an unstarred regular arg"""

        source = """
x = f(1, 2, 3)
"""
        expected = """
a6 = 1
a3 = [a6]
a9 = 2
a7 = [a9]
a2 = a3 + a7
a8 = 3
a4 = [a8]
r1 = a2 + a4
r5 = {}
x = f(*r1, **r5)
"""

        self.check_rewrite(source, expected)

    def disabled_test_crashing_case(self) -> None:
        """Debugging a crash in an external test"""

        # PYTHON VERSIONING ISSUE
        # TODO: There is some difference in the parse trees in the new version of
        # Python that we are not expecting. Until we understand what is going on,
        # disable this test.

        source = """
def flip_logit_constant():
  return Bernoulli(logits=tensor(-2.0))
"""
        expected = """
def flip_logit_constant():
    r2 = []
    a7 = 2.0
    a6 = -a7
    r5 = [a6]
    r8 = {}
    a4 = tensor(*r5, **r8)
    r3 = dict(logits=a4)
    r1 = Bernoulli(*r2, **r3)
    return r1
"""
        self.check_rewrite(source, expected)

        self.check_rewrite_as_ast(source, expected)

    def test_single_assignment_listComp(self) -> None:
        """Test the assign rule for desugaring listComps"""
        # TODO: We should add some tests to check that we
        # handle nested function definitions correctly

        self.maxDiff = None

        source = """
x = [i for i in range(0,j) if even(i+j)]
"""
        expected = """
def p1():
    r2 = []
    for i in range(0, j):
        if even(i + j):
            r2.append(i)
    return r2


x = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_listComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_listComp()))
        )

        expected = """
def p1():
    r2 = []
    a15 = 0
    a12 = [a15]
    a16 = [j]
    r10 = a12 + a16
    r17 = {}
    f3 = range(*r10, **r17)
    for i in f3:
        a11 = i + j
        r7 = [a11]
        r13 = {}
        r4 = even(*r7, **r13)
        if r4:
            a8 = r2.append
            r14 = [i]
            r18 = {}
            u6 = a8(*r14, **r18)
    return r2


r5 = []
r9 = {}
x = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

        source = """
y = [(x,y) for x in range(0,10) for y in range (x,10) if y == 2*x]
"""
        expected = """
def p1():
    r2 = []
    for x in range(0, 10):
        for y in range(x, 10):
            if y == 2 * x:
                r2.append((x, y))
    return r2


y = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_listComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_listComp()))
        )

        expected = """
def p1():
    r2 = []
    a15 = 0
    a13 = [a15]
    a20 = 10
    a16 = [a20]
    r10 = a13 + a16
    r17 = {}
    f3 = range(*r10, **r17)
    for x in f3:
        a18 = [x]
        a24 = 10
        a21 = [a24]
        r14 = a18 + a21
        r22 = {}
        f4 = range(*r14, **r22)
        for y in f4:
            a11 = 2
            a7 = a11 * x
            r6 = y == a7
            if r6:
                a12 = r2.append
                a23 = x, y
                r19 = [a23]
                r25 = {}
                u8 = a12(*r19, **r25)
    return r2


r5 = []
r9 = {}
y = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

        source = """
y = [(x,y) for x in range(0,10) if x>0 for y in range (x,10) if y == 2*x]
"""
        expected = """
def p1():
    r2 = []
    for x in range(0, 10):
        if x > 0:
            for y in range(x, 10):
                if y == 2 * x:
                    r2.append((x, y))
    return r2


y = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_listComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_listComp()))
        )

        expected = """
def p1():
    r2 = []
    a16 = 0
    a13 = [a16]
    a20 = 10
    a17 = [a20]
    r10 = a13 + a17
    r18 = {}
    f3 = range(*r10, **r18)
    for x in f3:
        a6 = 0
        r4 = x > a6
        if r4:
            a21 = [x]
            a26 = 10
            a23 = [a26]
            r19 = a21 + a23
            r24 = {}
            f7 = range(*r19, **r24)
            for y in f7:
                a14 = 2
                a11 = a14 * x
                r8 = y == a11
                if r8:
                    a15 = r2.append
                    a25 = x, y
                    r22 = [a25]
                    r27 = {}
                    u12 = a15(*r22, **r27)
    return r2


r5 = []
r9 = {}
y = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_setComp(self) -> None:
        """Test the assign rule for desugaring setComps"""
        # TODO: We should add some tests to check that we
        # handle nested function definitions correctly

        self.maxDiff = None

        source = """
x = {i for i in range(0,j) if even(i+j)}
"""
        expected = """
def p1():
    r2 = set()
    for i in range(0, j):
        if even(i + j):
            r2.add(i)
    return r2


x = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_setComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_setComp()))
        )

        expected = """
def p1():
    r2 = set()
    a15 = 0
    a12 = [a15]
    a16 = [j]
    r10 = a12 + a16
    r17 = {}
    f3 = range(*r10, **r17)
    for i in f3:
        a11 = i + j
        r7 = [a11]
        r13 = {}
        r4 = even(*r7, **r13)
        if r4:
            a8 = r2.add
            r14 = [i]
            r18 = {}
            u6 = a8(*r14, **r18)
    return r2


r5 = []
r9 = {}
x = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

        source = """
y = {(x,y) for x in range(0,10) for y in range (x,10) if y == 2*x}
"""
        expected = """
def p1():
    r2 = set()
    for x in range(0, 10):
        for y in range(x, 10):
            if y == 2 * x:
                r2.add((x, y))
    return r2


y = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_setComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_setComp()))
        )

        expected = """
def p1():
    r2 = set()
    a15 = 0
    a13 = [a15]
    a20 = 10
    a16 = [a20]
    r10 = a13 + a16
    r17 = {}
    f3 = range(*r10, **r17)
    for x in f3:
        a18 = [x]
        a24 = 10
        a21 = [a24]
        r14 = a18 + a21
        r22 = {}
        f4 = range(*r14, **r22)
        for y in f4:
            a11 = 2
            a7 = a11 * x
            r6 = y == a7
            if r6:
                a12 = r2.add
                a23 = x, y
                r19 = [a23]
                r25 = {}
                u8 = a12(*r19, **r25)
    return r2


r5 = []
r9 = {}
y = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

        source = """
y = {(x,y) for x in range(0,10) if x>0 for y in range (x,10) if y == 2*x}
"""
        expected = """
def p1():
    r2 = set()
    for x in range(0, 10):
        if x > 0:
            for y in range(x, 10):
                if y == 2 * x:
                    r2.add((x, y))
    return r2


y = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_setComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_setComp()))
        )

        expected = """
def p1():
    r2 = set()
    a16 = 0
    a13 = [a16]
    a20 = 10
    a17 = [a20]
    r10 = a13 + a17
    r18 = {}
    f3 = range(*r10, **r18)
    for x in f3:
        a6 = 0
        r4 = x > a6
        if r4:
            a21 = [x]
            a26 = 10
            a23 = [a26]
            r19 = a21 + a23
            r24 = {}
            f7 = range(*r19, **r24)
            for y in f7:
                a14 = 2
                a11 = a14 * x
                r8 = y == a11
                if r8:
                    a15 = r2.add
                    a25 = x, y
                    r22 = [a25]
                    r27 = {}
                    u12 = a15(*r22, **r27)
    return r2


r5 = []
r9 = {}
y = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_dictComp(self) -> None:
        """Test the assign rule for desugaring dictComps"""
        # TODO: We should add some tests to check that we
        # handle nested function definitions correctly

        self.maxDiff = None

        source = """
x = {i:i for i in range(0,j) if even(i+j)}
"""
        expected = """
def p1():
    r2 = {}
    for i in range(0, j):
        if even(i + j):
            r2.__setitem__(i, i)
    return r2


x = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_dictComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_dictComp()))
        )

        expected = """
def p1():
    r2 = {}
    a14 = 0
    a12 = [a14]
    a15 = [j]
    r10 = a12 + a15
    r16 = {}
    f3 = range(*r10, **r16)
    for i in f3:
        a11 = i + j
        r7 = [a11]
        r13 = {}
        r4 = even(*r7, **r13)
        if r4:
            a8 = r2.__setitem__
            a18 = [i]
            a19 = [i]
            r17 = a18 + a19
            r20 = {}
            u6 = a8(*r17, **r20)
    return r2


r5 = []
r9 = {}
x = p1(*r5, **r9)
"""
        self.check_rewrite(source, expected)

        source = """
y = {x:y for x in range(0,10) for y in range (x,10) if y == 2*x}
"""
        expected = """
def p1():
    r2 = {}
    for x in range(0, 10):
        for y in range(x, 10):
            if y == 2 * x:
                r2.__setitem__(x, y)
    return r2


y = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_dictComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_dictComp()))
        )

        expected = """
def p1():
    r2 = {}
    a15 = 0
    a13 = [a15]
    a19 = 10
    a16 = [a19]
    r10 = a13 + a16
    r17 = {}
    f3 = range(*r10, **r17)
    for x in f3:
        a18 = [x]
        a22 = 10
        a20 = [a22]
        r14 = a18 + a20
        r21 = {}
        f4 = range(*r14, **r21)
        for y in f4:
            a11 = 2
            a7 = a11 * x
            r6 = y == a7
            if r6:
                a12 = r2.__setitem__
                a24 = [x]
                a25 = [y]
                r23 = a24 + a25
                r26 = {}
                u8 = a12(*r23, **r26)
    return r2


r5 = []
r9 = {}
y = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

        source = """
y = {x:y for x in range(0,10) if x>0 for y in range (x,10) if y == 2*x}
"""
        expected = """
def p1():
    r2 = {}
    for x in range(0, 10):
        if x > 0:
            for y in range(x, 10):
                if y == 2 * x:
                    r2.__setitem__(x, y)
    return r2


y = p1()
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_dictComp())
        )

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_assign_dictComp()))
        )

        expected = """
def p1():
    r2 = {}
    a16 = 0
    a13 = [a16]
    a20 = 10
    a17 = [a20]
    r10 = a13 + a17
    r18 = {}
    f3 = range(*r10, **r18)
    for x in f3:
        a6 = 0
        r4 = x > a6
        if r4:
            a21 = [x]
            a24 = 10
            a22 = [a24]
            r19 = a21 + a22
            r23 = {}
            f7 = range(*r19, **r23)
            for y in f7:
                a14 = 2
                a11 = a14 * x
                r8 = y == a11
                if r8:
                    a15 = r2.__setitem__
                    a26 = [x]
                    a27 = [y]
                    r25 = a26 + a27
                    r28 = {}
                    u12 = a15(*r25, **r28)
    return r2


r5 = []
r9 = {}
y = p1(*r5, **r9)
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_nested_call_named_arg(self) -> None:
        self.maxDiff = None

        # This test originally pointed out a bug in the rewriting logic.
        # We should be pulling the invocation of c() out into
        # it's own top-level function call.
        #
        # The code below should be rewritten as something like:
        #
        # t1 = []
        # t2 = {}
        # t3 = c(*t1, **t2)
        # t4 = []
        # t5 = {'n' : t3}
        # t6 = b(*t4, **t5)
        # return t6

        source = """
def f():
    return b(n=c())
"""
        expected = """
def f():
    r2 = []
    r3 = dict(n=c())
    r1 = b(*r2, **r3)
    return r1
"""
        # The previous "expected" was the undesirable output, which we got at the time of the bug report
        # The following "expected" is after the bug fix
        expected = """
def f():
    r2 = []
    r5 = []
    r6 = {}
    a4 = c(*r5, **r6)
    r3 = dict(n=a4)
    r1 = b(*r2, **r3)
    return r1
"""

        self.check_rewrite(source, expected)

        # Helper tests to fix the bug identified above
        # Interestingly, regular arguments are OK

        source = """
def f():
    return b(c())
"""
        expected = """
def f():
    r5 = []
    r6 = {}
    a3 = c(*r5, **r6)
    r2 = [a3]
    r4 = {}
    r1 = b(*r2, **r4)
    return r1
"""

        self.check_rewrite(source, expected)

        # It was further noted that the following expression was
        # also not handled well

        source = """
def f():
    return b(n1=c1(),n2=c2(),n3=c3())
"""

        # In particular, it produced the following output, which
        # has nested "dict" calls that are should be removed
        expected = """
def f():
    r2 = []
    r3 = dict(**dict(**dict(n1=c1()), **dict(n2=c2())), **dict(n3=c3()))
    r1 = b(*r2, **r3)
    return r1
"""
        # To fix this, first we introduced the rewrite "binary_dict_left"
        # With the introduction of that rule we get
        expected = """
def f():
    r2 = []
    r7 = []
    r8 = {}
    a6 = c1(*r7, **r8)
    a5 = dict(n1=a6)
    a4 = dict(**a5, **dict(n2=c2()))
    r3 = dict(**a4, **dict(n3=c3()))
    r1 = b(*r2, **r3)
    return r1
"""
        # Next, we introduced "binary_dict_right" and then we get
        expected = """
def f():
    r2 = []
    r11 = []
    r14 = {}
    a7 = c1(*r11, **r14)
    a5 = dict(n1=a7)
    r13 = []
    r16 = {}
    a10 = c2(*r13, **r16)
    a8 = dict(n2=a10)
    a4 = dict(**a5, **a8)
    r12 = []
    r15 = {}
    a9 = c3(*r12, **r15)
    a6 = dict(n3=a9)
    r3 = dict(**a4, **a6)
    r1 = b(*r2, **r3)
    return r1
"""

        self.check_rewrite(source, expected)

        # It was useful to note that there was no similar problem with
        # calls that have regular arguments

        source = """
def f():
    return b(*[c()])
"""
        expected = """
def f():
    r5 = []
    r6 = {}
    a3 = c(*r5, **r6)
    r2 = [a3]
    r4 = {}
    r1 = b(*r2, **r4)
    return r1
"""

        self.check_rewrite(source, expected)

        # No similar problem with multiple regular arguments also:

        source = """
def f():
    return b(c1(),c2())
"""
        expected = """
def f():
    r8 = []
    r10 = {}
    a4 = c1(*r8, **r10)
    a3 = [a4]
    r9 = []
    r11 = {}
    a7 = c2(*r9, **r11)
    a5 = [a7]
    r2 = a3 + a5
    r6 = {}
    r1 = b(*r2, **r6)
    return r1
"""

        self.check_rewrite(source, expected)

    def test_single_assignment_assign_unary_dict(self) -> None:
        """Test the first special rule for dict (the unary case)"""

        self.maxDiff = None

        source = """
x = dict(n=c())
"""
        expected = """
a1 = c()
x = dict(n=a1)
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_unary_dict())
        )

    def test_single_assignment_assign_binary_dict_left(self) -> None:
        """Test the first special rule for dict (the binary left case)"""

        self.maxDiff = None

        source = """
x = dict(**c(),**d())
"""
        expected = """
a1 = c()
x = dict(**a1, **d())
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_binary_dict_left())
        )

    def test_single_assignment_assign_binary_dict_right(self) -> None:
        """Test the first special rule for dict (the binary right case)"""

        self.maxDiff = None

        source = """
x = dict(**c,**d())
"""
        expected = """
a1 = d()
x = dict(**c, **a1)
"""

        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_assign_binary_dict_right())
        )

    def test_left_value_all(self) -> None:
        """General tests for the full set of assignment left value rules"""
        # First, some "most general" normal forms. These are terms that are not
        # reduced by this set of rewrites, nor by all the other rules for that matter.
        normal_forms = [
            """
def f(x):
    a = z
    a.b = z
    a[b] = z
    a[b:c] = z
    a[b:] = z
    a[:c] = z
    a[b:c:d] = z
    a[b::d] = z
    a[:c:d] = z
    a[::d] = z
    [] = z
    [a] = z
    [*a] = z
        """
        ]

        # These terms are normal forms for this specific set
        self.check_rewrites(normal_forms, self.s._handle_left_value_all())
        # They are also "in most general form" because they are also normal forms for all sets
        self.check_rewrites(normal_forms)
        # It would be nice of course if we could check that we have captured (at least
        # representatives) of all normal form productions, but no idea how to do this yet.

        # Second, some terms that are only in normal form for this set (but could be
        # reducible by other rules). This type of terms helps us check the rules in
        # this set do not rewrite terms prematurely (which could alter order of evaluation).
        # Note: It's good for such terms to actually contain a reduction that can be done
        # once the subterm that is "waited upon" is released. This means that if we want
        # to systematically derive waiting terms from normal forms, two subterms would
        # typically need to be changed.

        waiting_forms = [
            """
def f(x):
    x.y.a = z + 1
    x.y[b] = z + 1
    a[x.y] = z + 1
    x.y[b:c] = z + 1
    x.y[b:] = z + 1
    x.y[:c] = z + 1
    a[x.y:c] = z + 1
    a[x.y:] = z + 1
    a[b:x.y] = z + 1
    a[:x.y] = z + 1
    x.y[:c:d] = z + 1
    x.y[b::d] = z + 1
    x.y[::d] = z + 1
    x.y[b:c:d] = z + 1
    a[x.y:c:d] = z + 1
    a[x.y::d] = z + 1
    a[x.y:] = z + 1
    a[b:x.y:d] = z + 1
    a[b:x.y:d] = z + 1
    a[b:x.y] = z + 1
    a[:x.y:d] = z + 1
    a[:c:x.y] = z + 1
    a[::x.y] = z + 1
    [x.y.a] = z + 1
    [*x.y.a] = z + 1
        """
        ]

        self.check_rewrites(waiting_forms, self.s._handle_left_value_all())

    def test_left_value_attributeref(self) -> None:
        """Test rewrites like a.b.c = z → x = a.b; x.c = z"""

        terms = [
            """
def f(x):
    a.b.c = z""",
            """
def f(x):
    x1 = a.b
    x1.c = z""",
        ]

        self.check_rewrites(terms, self.s._handle_left_value_attributeref())
        self.check_rewrites(terms, self.s._handle_left_value_all())
        self.check_rewrites(terms)

    def test_left_value_subscript_value(self) -> None:
        """Test rewrites like a.b[c] = z → x = a.b; x[c] = z.
        It also handles [c], [c:d], and [c:d:e] in the same way."""

        terms = [
            """
def f(x):
    a.b[c] = z""",
            """
def f(x):
    x1 = a.b
    x1[c] = z""",
        ]

        self.check_rewrites(terms, self.s._handle_left_value_subscript_value())
        self.check_rewrites(terms, self.s._handle_left_value_all())
        self.check_rewrites(terms)

        terms = [
            """
def f(x):
    a.b[c:d] = z""",
            """
def f(x):
    x1 = a.b
    x1[c:d] = z""",
        ]

        self.check_rewrites(terms, self.s._handle_left_value_subscript_value())
        self.check_rewrites(terms, self.s._handle_left_value_all())
        self.check_rewrites(terms)

        terms = [
            """
def f(x):
    a.b[c:d:e] = z""",
            """
def f(x):
    x1 = a.b
    x1[c:d:e] = z""",
        ]

        self.check_rewrites(terms, self.s._handle_left_value_subscript_value())
        self.check_rewrites(terms, self.s._handle_left_value_all())
        self.check_rewrites(terms)

    def test_left_value_subscript_slice_index(self) -> None:
        """Test rewrites like a[b.c] = z → x = b.c; a[x] = z."""

        terms = [
            """
def f(x):
    a[b.c] = z""",
            """
def f(x):
    x1 = b.c
    a[x1] = z""",
        ]

        self.check_rewrites(terms, self.s._handle_left_value_subscript_slice_index())
        self.check_rewrites(terms, self.s._handle_left_value_all())
        self.check_rewrites(terms)

    def test_left_value_subscript_slice_lower(self) -> None:
        """Test rewrites like a[b.c:] = z → x = b.c; a[x:] = z."""

        terms = [
            """
def f(x):
    a[b.c:] = z
    a[b.c:d] = z
    a[b.c:d:e] = z
    a[:d:e] = z""",
            """
def f(x):
    x1 = b.c
    a[x1:] = z
    x2 = b.c
    a[x2:d] = z
    x3 = b.c
    a[x3:d:e] = z
    a[:d:e] = z""",
        ]

        self.check_rewrites(terms, self.s._handle_left_value_subscript_slice_lower())
        self.check_rewrites(terms, self.s._handle_left_value_all())
        self.check_rewrites(terms)
