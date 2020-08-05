# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for single_assignment.py"""
import ast
import unittest

import astor
from beanmachine.ppl.utils.ast_patterns import ast_domain
from beanmachine.ppl.utils.fold_constants import fold
from beanmachine.ppl.utils.rules import FirstMatch as first, TryMany as many
from beanmachine.ppl.utils.single_assignment import SingleAssignment


_some_top_down = ast_domain.some_top_down


class SingleAssignmentTest(unittest.TestCase):

    s = SingleAssignment()
    default_rules = s._rules

    def test_single_assignment_null_check(self) -> None:
        """To check if you change one of the two numbers the test fails"""

        self.assertEqual(3, 3)

    def test_single_assignment_unique_id(self) -> None:
        """Tests unique_uses intended preserves name prefix"""

        s = SingleAssignment()
        root = "root"
        name = s._unique_id(root)
        self.assertEqual(root, name[0 : len(root)])

    def check_rewrite(self, source, expected, rules=default_rules):
        """Tests that starting from applying rules to source yields expected"""

        self.s._count = 0
        self.s._rules = rules
        m = ast.parse(source)
        result = self.s.single_assignment(fold(m))
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def check_rewrite_ast(self, source, expected, rules=default_rules):
        """Tests that starting from applying rules to source yields expected ast"""

        self.maxDiff = None
        self.s._count = 0
        m = ast.parse(source)
        result = self.s.single_assignment(fold(m))
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))

    def test_single_assignment_pre_unassigned_expressions(self) -> None:
        """Tests need for a new rule to handle unassigned expressions"""

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
    while not(True):
        x=x+1
    else:
        x=x-1
"""
        expected = """
def f(x):
    while True:
        w1 = False
        if w1:
            x = x + 1
    if not w1:
        x = x - 1
"""
        self.check_rewrite(
            source, expected, _some_top_down(self.s._handle_while_not_True())
        )

        # Check that the while_not_True rewrite reaches normal form
        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_while_not_True()))
        )

        # Check that while_True rule works (alone) on simple cases

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
            source, expected, _some_top_down(self.s._handle_while_True())
        )

        # Check that while_True rule, alone, on simple cases, reaches a normal form

        self.check_rewrite(
            source, expected, many(_some_top_down(self.s._handle_while_True()))
        )

        # Check that (combined) while rule works (alone) on simple cases

        source = """
def f(x):
    while not(True):
        x=x+1
    else:
        x=x-1
    while True:
        y=y+1
    else:
        y=y-1

"""
        expected = """
def f(x):
    while True:
        w1 = False
        if w1:
            x = x + 1
    if not w1:
        x = x - 1
    while True:
        y = y + 1
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
    while not(True):
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
        w1 = False
        if w1:
            a5 = 1
            a2 = x + a5
            x = a2 - s
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
        r26 = dict(y=6)
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
r1 = dict(**dict(**a, **b), **c)
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
r1 = dict(**dict(**dict(**d), **dict(k=42)), **dict(**e))
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
r2 = dict(**dict(**dict(**d), **dict(k=42)), **dict(**e))
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

    def test_crashing_case(self) -> None:
        """Debugging a crash in an external test"""

        source = """
def flip_logit_constant():
  return Bernoulli(logits=tensor(-2.0))
"""
        expected = """
def flip_logit_constant():
    r2 = []
    r3 = dict(logits=tensor(-2.0))
    r1 = Bernoulli(*r2, **r3)
    return r1
"""
        self.check_rewrite(source, expected)

        self.check_rewrite_ast(source, expected)
