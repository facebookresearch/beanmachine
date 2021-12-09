#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A rules engine for tree transformation"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Tuple

from beanmachine.ppl.compiler.patterns import (
    Pattern,
    anyPattern,
    failPattern,
    is_any,
    match,
    to_pattern,
)


# Logically, a rule is just a projection; it's a partial function from
# any value to any other value.
#
# Since rules are partial functions -- they are allowed to reject their
# argument and fail -- we will implement rules as classes with an apply
# method that returns a success or failure code.
#
# There are a number of ways to construct rules; a basic way is to
# provide a pattern -- a predicate on values -- and an action to take
# if the pattern is matched successfully -- that is, a function from
# values to values.
#
# Rules may be combined together with *rule combinators*; a combinator is
# a function which takes one or more rules and produces a rule.


_empty = {}


class RuleResult(ABC):
    test: Any

    def __init__(self, test: Any) -> None:
        self.test = test

    @abstractmethod
    def is_success(self) -> bool:
        pass

    @abstractmethod
    def is_fail(self) -> bool:
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__}:{self.test}"

    @abstractmethod
    def expect_success(self) -> Any:
        pass

    def __bool__(self) -> bool:
        return self.is_success()


class Fail(RuleResult):
    def __init__(self, test: Any = None) -> None:
        RuleResult.__init__(self, test)

    def is_success(self) -> bool:
        return False

    def is_fail(self) -> bool:
        return True

    def expect_success(self) -> Any:
        raise ValueError("Expected success but rewrite rule patten match failed")


class Success(RuleResult):
    result: Any

    def __init__(self, test: Any, result: Any) -> None:
        RuleResult.__init__(self, test)
        self.result = result

    def is_success(self) -> bool:
        return True

    def is_fail(self) -> bool:
        return False

    def expect_success(self) -> Any:
        return self.result


class Rule(ABC):
    """A rule represents a partial function that transforms a value."""

    name: str

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def apply(self, test: Any) -> RuleResult:
        pass

    def __call__(self, test: Any) -> RuleResult:
        return self.apply(test)

    @abstractmethod
    def always_succeeds(self) -> bool:
        pass


def _identity(x: Any) -> Any:
    return x


class Trace(Rule):
    """This combinator introduces a side effect to be executed every time the
    child rule is executed, and when it succeeds or fails. It is useful for
    debugging."""

    rule: Rule
    logger: Callable[[Rule, Any], None]

    def __init__(self, rule: Rule, logger: Callable[[Rule, Any], None]) -> None:
        Rule.__init__(self, rule.name)
        self.rule = rule
        self.logger = logger

    def apply(self, test: Any) -> RuleResult:
        self.logger(self.rule, None)
        result = self.rule(test)
        self.logger(self.rule, result)
        return result

    def __str__(self) -> str:
        return str(self.rule)

    def always_succeeds(self) -> bool:
        return self.rule.always_succeeds()


def make_logger(log: List[str]) -> Callable[[Rule], Rule]:
    def logger(rule: Rule, value: Any) -> None:
        if value is None:
            log.append(f"Started {rule.name}")
        else:
            log.append(f"Finished {rule.name}")

    def trace(rule: Rule) -> Rule:
        return Trace(rule, logger)

    return trace


class PatternRule(Rule):
    """If the test value matches the pattern, then the test value is passed
    to the projection and the rule succeeds. Otherwise, the rule fails."""

    pattern: Pattern
    projection: Callable[[Any], Any]

    def __init__(
        self,
        pattern: Pattern,
        projection: Callable[[Any], Any] = _identity,
        name: str = "pattern",
    ) -> None:
        Rule.__init__(self, name)
        self.pattern = pattern
        self.projection = projection

    def apply(self, test: Any) -> RuleResult:
        match_result = match(self.pattern, test)
        if match_result.is_fail():
            return Fail(test)
        result = self.projection(test)
        return Success(test, result)

    def __str__(self) -> str:
        return f"{self.name}( {str(to_pattern(self.pattern)) }"

    def always_succeeds(self) -> bool:
        return is_any(self.pattern)


def projection_rule(projection: Callable[[Any], Any], name: str = "projection") -> Rule:
    return PatternRule(anyPattern, projection, name)


# The identity rule is the rule that always succeeds, and the projection
# is an identity function.
identity: Rule = projection_rule(_identity, "identity")
# The fail rule is the rule that never succeeds.
fail: Rule = PatternRule(failPattern, _identity, "fail")

# This rule succeeds if the test is a list.
is_list: Rule = PatternRule(list, _identity, "is_list")


def always_replace(value: Any, name: str = "always_replace") -> Rule:
    """always_replace(value) produces a rule that replaces anything with
    the given value. It always succeeds."""
    return projection_rule(lambda x: value, name)


def pattern_rules(
    pairs: List[Tuple[Pattern, Callable[[Any], Any]]], name: str = "pattern_rules"
) -> Rule:
    """Constructs a rule from a sequence of pairs of patterns and projections.
    Patterns are checked in order, and the first one that matches is used for the
    projection; if none match then the rule fails."""
    rules = (PatternRule(pattern, action, name) for pattern, action in pairs)
    return FirstMatch(rules)


_exception = [Exception]


class IgnoreException(Rule):
    """Apply the given rule; if it throws an exception, the rule fails."""

    rule: Rule
    expected: List[type]

    def __init__(
        self, rule: Rule, expected: List[type] = _exception, name: str = "handle"
    ) -> None:
        Rule.__init__(self, name)
        self.rule = rule
        self.expected = expected

    def apply(self, test: Any) -> RuleResult:
        try:
            return self.rule.apply(test)
        except Exception as x:
            if any(isinstance(x, t) for t in self.expected):
                return Fail(test)
            # We did not expect this exception; do not eat the bug.
            raise

    def __str__(self) -> str:
        r = str(self.rule)
        return f"ignore_exception( {r} )"

    def always_succeeds(self) -> bool:
        # Presumably you would not be wrapping a rule that never throws,
        # so let's assume that this can fail.
        return False


def ignore_div_zero(rule: Rule) -> Rule:
    return IgnoreException(rule, [ZeroDivisionError], "ignore_div_zero")


def ignore_runtime_error(rule: Rule) -> Rule:
    return IgnoreException(rule, [RuntimeError], "ignore_runtime_error")


def ignore_value_error(rule: Rule) -> Rule:
    return IgnoreException(rule, [ValueError], "ignore_value_error")


class Check(Rule):
    """Apply the given rule; if it fails, fail. If it succeeds, the result
    is the original test value, not the transformed value.  This is useful
    for scenarios where we wish to know if a particular thing is true of
    a node before we apply an expensive rule to it."""

    rule: Rule

    def __init__(self, rule: Rule, name: str = "check") -> None:
        Rule.__init__(self, name)
        self.rule = rule

    def apply(self, test: Any) -> RuleResult:
        rule_result = self.rule.apply(test)
        if rule_result.is_success():
            return Success(test, test)
        return rule_result

    def __str__(self) -> str:
        r = str(self.rule)
        return f"check( {r} )"

    def always_succeeds(self) -> bool:
        # Note that it is strange to have a Check which always succeeds
        # because that is the same as the identity rule.
        # TODO: Consider implementing some sort of warning for this case?
        return self.rule.always_succeeds()


class Choose(Rule):
    """Apply the condition rule to the test.
    If it succeeds, apply the rule in the consequence to its output.
    If it fails, apply the rule in the alternative to the test.
    That is, Choose(a, b, c)(test) has the semantics of
    if a(test) then b(a(test)) else c(test)"""

    condition: Rule
    consequence: Rule
    alternative: Rule

    def __init__(
        self,
        condition: Rule,
        consequence: Rule,
        alternative: Rule,
        name: str = "choose",
    ) -> None:
        Rule.__init__(self, name)
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative

    def apply(self, test: Any) -> RuleResult:
        rule_result = self.condition.apply(test)
        if isinstance(rule_result, Success):
            return self.consequence.apply(rule_result.result)
        return self.alternative.apply(test)

    def __str__(self) -> str:
        a = str(self.condition)
        b = str(self.consequence)
        c = str(self.alternative)
        return f"choose( {a}, {b}, {c} )"

    def always_succeeds(self) -> bool:
        if self.condition.always_succeeds():
            return self.consequence.always_succeeds()
        return self.consequence.always_succeeds() and self.alternative.always_succeeds()


def if_then(condition: Rule, consequence: Rule, alternative: Rule = identity) -> Rule:
    """Apply the condition rule, then apply the original test to either the
    consequence or the alternative, depending on whether the condition succeeded
    or failed. Note that this is different than Choose. Choose applies the
    condition to the result of the condition, not to the original test."""
    return Choose(Check(condition), consequence, alternative)


class Compose(Rule):
    """Apply the first rule to the test.
    If it succeeds, apply the second rule to its output.
    That is, Compose(a, b)(test) has the semantics of
    if a(test) then b(a(test)) else fail"""

    # Compose could be implemented as Choose(a, b, fail), but for debugging
    # purposes it is better to explicitly implement it.

    first: Rule
    second: Rule

    def __init__(self, first: Rule, second: Rule, name: str = "compose") -> None:
        Rule.__init__(self, name)
        self.first = first
        self.second = second

    def apply(self, test: Any) -> RuleResult:
        rule_result = self.first.apply(test)
        if isinstance(rule_result, Success):
            return self.second.apply(rule_result.result)
        return rule_result

    def __str__(self) -> str:
        a = str(self.first)
        b = str(self.second)
        return f"compose( {a}, {b} )"

    def always_succeeds(self) -> bool:
        return self.first.always_succeeds() and self.second.always_succeeds()


class Recursive(Rule):
    """Delay construction of a rule until we need it, so as to avoid recursion."""

    rule_maker: Callable[[], Rule]

    def __init__(self, rule_maker: Callable[[], Rule], name: str = "recursive") -> None:
        Rule.__init__(self, name)
        self.rule_maker = rule_maker

    def apply(self, test: Any) -> RuleResult:
        return self.rule_maker().apply(test)

    def __str__(self) -> str:
        return self.name

    def always_succeeds(self) -> bool:
        return False


class OrElse(Rule):
    """Apply the first rule to the test.
    If it succeeds, use that result.
    If it fails, apply the second rule to the test and return that."""

    # OrElse could be implemented as Choose(first, identity, second), but for debugging
    # purposes it is better to explicitly implement it.

    first: Rule
    second: Rule

    def __init__(self, first: Rule, second: Rule, name: str = "or_else") -> None:
        Rule.__init__(self, name)
        self.first = first
        self.second = second

    def apply(self, test: Any) -> RuleResult:
        rule_result = self.first.apply(test)
        if isinstance(rule_result, Success):
            return rule_result
        return self.second.apply(test)

    def __str__(self) -> str:
        a = str(self.first)
        b = str(self.second)
        return f"or_else( {a}, {b} )"

    def always_succeeds(self) -> bool:
        return self.first.always_succeeds() or self.second.always_succeeds()


class FirstMatch(Rule):
    """Apply each rule to the test until one succeeds; if none succeed, then fail."""

    # FirstMatch([a,b,c]) could be implemented as OrElse(a, OrElse(b, c)) but for
    # debugging purposes it is better to explicitly implement it.

    rules: List[Rule]

    def __init__(self, rules: Iterable[Rule], name: str = "first_match") -> None:
        Rule.__init__(self, name)
        self.rules = list(rules)

    def apply(self, test: Any) -> RuleResult:
        for rule in self.rules:
            rule_result = rule.apply(test)
            if isinstance(rule_result, Success):
                return rule_result
        return Fail(test)

    def __str__(self) -> str:
        rs = ", ".join(str(rule) for rule in self.rules)
        return f"first_match( {rs} )"

    def always_succeeds(self) -> bool:
        return any(r.always_succeeds() for r in self.rules)


class TryOnce(Rule):
    """Apply the rule to the test.
    If it succeeds, use that result.
    If it fails, use the test as the result and succeed.
    This rule always succeeds."""

    # TryOnce could be implemented as OrElse(rule, identity), but for debugging
    # purposes it is better to explicitly implement it.

    rule: Rule

    def __init__(self, rule: Rule, name: str = "try_once") -> None:
        Rule.__init__(self, name)
        self.rule = rule

    def apply(self, test: Any) -> RuleResult:
        rule_result = self.rule.apply(test)
        if isinstance(rule_result, Success):
            return rule_result
        return Success(test, test)

    def __str__(self) -> str:
        return f"try_once( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return True


def either_or_both(first: Rule, second: Rule, name: str = "either_or_both") -> Rule:
    """Do the first rule; if it succeeds, try doing the second rule, but do
    not worry if it fails. If the first rule fails, do the second rule. The
    net effect is, either first, or second, or first-then-second happens,
    or both fail."""
    return Choose(first, TryOnce(second), second, name)


class SomeOf(Rule):
    # This is logically the extension of either-or-both to arbitrarily many rules.
    """Takes a list of rules and composes together as many of them as succeed.
    At least one must succeed, otherwise the rule fails."""
    rules: List[Rule]

    def __init__(self, rules: List[Rule], name: str = "some_of") -> None:
        Rule.__init__(self, name)
        self.rules = rules

    def apply(self, test: Any) -> RuleResult:
        result = Fail()
        current_test = test
        for current_rule in self.rules:
            current_result = current_rule.apply(current_test)
            # If we succeeded, this becomes the input to the next rule.
            # If we failed, just ignore it and try the next rule.
            if current_result.is_success():
                current_test = current_result.expect_success()
                result = current_result
        if result.is_success():
            return Success(test, result.expect_success())
        return Fail(test)

    def __str__(self) -> str:
        rs = ",".join(str(r) for r in self.rules)
        return f"some_of( {rs} )"

    def always_succeeds(self) -> bool:
        return any(r.always_succeeds() for r in self.rules)


class AllOf(Rule):
    # This is logically the extension of composition to arbitrarily many rules.
    """Takes a list of rules and composes together all of them.
    All must succeed, otherwise the rule fails."""
    rules: List[Rule]

    def __init__(self, rules: List[Rule], name: str = "all_of") -> None:
        Rule.__init__(self, name)
        self.rules = rules

    def apply(self, test: Any) -> RuleResult:
        current_test = test
        result = Success(test, test)
        for current_rule in self.rules:
            result = current_rule.apply(current_test)
            if result.is_fail():
                return Fail(test)
            current_test = result.expect_success()
        return Success(test, result.expect_success())

    def __str__(self) -> str:
        rs = ",".join(str(r) for r in self.rules)
        return f"all_of( {rs} )"

    def always_succeeds(self) -> bool:
        return all(r.always_succeeds() for r in self.rules)


class TryMany(Rule):
    """Repeatedly apply a rule; the result is that of the last application
    that succeeded, or the original test if none succeeded.
    This rule always succeeds."""

    # TryMany could be implemented as TryOnce(Compose(rule, Recursive(TryMany(rule))))
    # but for debugging purposes it is better to explicitly implement it.

    rule: Rule

    def __init__(self, rule: Rule, name: str = "try_many") -> None:
        Rule.__init__(self, name)
        self.rule = rule
        if rule.always_succeeds():
            raise ValueError(
                "TryMany has been given a rule that always succeeds,"
                + " which will cause an infinite loop."
            )

    def apply(self, test: Any) -> RuleResult:
        current: Success = Success(test, test)
        while True:
            rule_result = self.rule.apply(current.result)
            if isinstance(rule_result, Success):
                current = rule_result
            else:
                return current

    def __str__(self) -> str:
        return f"try_many( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return True


def at_least_once(rule: Rule) -> Rule:
    """Try a rule once; if it fails, fail. If it succeeds,
    try it again as many times as it keeps succeeding."""
    return Compose(rule, TryMany(rule))


class ListEdit:
    """Consider a rule which descends through an AST looking for a particular
    statement to replace. If the rule replaces a particular statement with
    another statement, we can express that with a straightforward rule that
    succeeds and produces the new statement. But how can we represent rules that
    either delete a statement (that is, replace it with nothing) or replace it
    with more than one statement?
    To express this concept, a rule should succeed and return a ListEdit([s1, s2...])
    where the list contains the replacements; if the list is empty then the
    element is deleted."""

    edits: List[Any]

    def __init__(self, edits: List[Any]) -> None:
        self.edits = edits


remove_from_list = ListEdit([])


def _expand_edits(items: Iterable[Any]) -> Iterable[Any]:
    """Suppose we have a list [X, Y, Z] and we wish to replace
    Y with A, B.  We will produce the sequence
    [X, ListEdit([A, B], Z)]; this function expands the ListEdit
    structure and splices the elements into the list, producing
    [X, A, B, Z]."""
    for item in items:
        if isinstance(item, ListEdit):
            for expanded in _expand_edits(item.edits):
                yield expanded
        else:
            yield item


def _list_unchanged(xs: List[Any], ys: List[Any]) -> bool:
    if xs is ys:
        return True
    # When we do a rewrite step that produces no change, we try to
    # guarantee that the "rewritten" value is reference identical to
    # the original value. We can therefore take advantage of this
    # fact when comparing two reference-unequal lists.  Using normal
    # structural equality on lists verifies that each member has the
    # same structure as the corresponding member, but we can be faster
    # than that by verifying that each member is reference equal; if
    # any member is reference-unequal then something was rewritten
    # to a different value.
    if len(xs) != len(ys):
        return False
    return all(x is y for x, y in zip(xs, ys))


class AllListMembers(Rule):
    """Apply a rule to all members.  Succeeds if the rule succeeds for all members,
    and returns a list with the members replaced with the new values.
    Otherwise, fails."""

    rule: Rule

    def __init__(self, rule: Rule, name: str = "all_list_members") -> None:
        Rule.__init__(self, name)
        self.rule = rule

    def apply(self, test: Any) -> RuleResult:
        # Easy outs:
        if not isinstance(test, list):
            return Fail(test)
        if len(test) == 0:
            return Success(test, test)
        results = [self.rule.apply(child) for child in test]
        # Were there any failures?
        if any(result.is_fail() for result in results):
            return Fail(test)
        # Splice in any list edits.
        new_values = list(_expand_edits(result.expect_success() for result in results))
        # Is the resulting list different? If not, make sure the result is
        # reference equal.
        if _list_unchanged(new_values, test):
            return Success(test, test)
        # Everything succeeded but there was at least one different value.
        return Success(test, new_values)

    def __str__(self) -> str:
        return f"all_list_members( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return False


# We have n-ary operators that have both "term" children and "list member"
# children; for example Compare(left, ops, comps) has a left child and then
# a list of arbitrarily many other operands in comps. When applying a rule to
# all children normally Compare is considered to have 3 children and we
# apply the rule to each, but it is convenient to apply the rule to all the
# logical children -- the ops, the comps, and left -- rather than to the lists
# themselves. This combinator enables that scenario.
# TODO: always_succeeds will be wrong for the returned object.
def list_member_children(rule: Rule) -> Rule:
    return if_then(is_list, AllListMembers(rule), rule)


class AllListEditMembers(Rule):
    """Rules which are intended to modify a parent list by adding or removing items
    return a ListEdit([...]) object, but in cases where a rule then recursese
    upon children -- like top_down -- we'll potentially need to rewrite the elements
    in edit list. This combinator implements that."""

    # The implementation strategy here is to just defer to AllListMembers for
    # the heavy lifting.

    rule: AllListMembers

    def __init__(self, rule: Rule, name: str = "all_list_edit_members") -> None:
        Rule.__init__(self, name)
        self.rule = AllListMembers(rule)

    def apply(self, test: Any) -> RuleResult:
        if not isinstance(test, ListEdit):
            return Fail(test)
        result = self.rule(test.edits)
        if result.is_fail():
            return Fail(test)
        new_values = result.expect_success()
        if new_values is test.edits:
            return Success(test, test)
        return Success(test, ListEdit(new_values))

    def __str__(self) -> str:
        return f"all_list_edit_members( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return False


class AllTermChildren(Rule):
    """Apply a rule to all children.  Succeeds if the rule succeeds for all children,
    and returns a constructed object with the children replaced with the new values.
    Otherwise, fails."""

    get_children: Callable[[Any], Dict[str, Any]]
    construct: Callable[[type, Dict[str, Any]], Any]
    rule: Rule

    def __init__(
        self,
        rule: Rule,
        get_children: Callable[[Any], Dict[str, Any]],
        construct: Callable[[type, Dict[str, Any]], Any],
        name: str = "all_children",
    ) -> None:
        Rule.__init__(self, name)
        self.rule = rule
        self.get_children = get_children
        self.construct = construct

    def apply(self, test: Any) -> RuleResult:
        children = self.get_children(test)
        # Easy out for leaves.
        if len(children) == 0:
            return Success(test, test)
        results = {
            child_name: self.rule.apply(child_value)
            for child_name, child_value in children.items()
        }
        # Were there any failures?
        if any(result.is_fail() for result in results.values()):
            return Fail(test)
        # Were there any successes that returned a different value?
        new_values = {n: results[n].expect_success() for n in results}
        if all(new_values[n] is children[n] for n in new_values):
            # Everything succeeded and there were no changes.
            return Success(test, test)
        # Everything succeeded but there was at least one different value.
        # Construct a new object.
        return Success(test, self.construct(type(test), new_values))

    def __str__(self) -> str:
        return f"all_term_children( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return self.rule.always_succeeds()


# The wrapper is just to ensure that always_succeeds has the right semantics.
class AllChildren(Rule):
    """Apply a rule to all children or list members."""

    rule: Rule
    combined_rule: Rule

    def __init__(
        self,
        rule: Rule,
        get_children: Callable[[Any], Dict[str, Any]],
        construct: Callable[[type, Dict[str, Any]], Any],
        name: str = "all_children",
    ) -> None:
        Rule.__init__(self, name)
        self.rule = rule
        self.combined_rule = FirstMatch(
            [
                AllListMembers(rule),
                AllListEditMembers(rule),
                AllTermChildren(rule, get_children, construct),
            ]
        )

    def apply(self, test: Any) -> RuleResult:
        return self.combined_rule(test)

    def __str__(self) -> str:
        return f"all_children( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return self.rule.always_succeeds()


class SomeListMembers(Rule):
    """Apply a rule to all members.  Succeeds if the rule succeeds for one or
    more members, and returns a list with the children replaced
    with the new values. Otherwise, fails."""

    rule: Rule

    def __init__(self, rule: Rule, name: str = "some_list_members") -> None:
        Rule.__init__(self, name)
        self.rule = rule

    def apply(self, test: Any) -> RuleResult:
        # Easy outs:
        if not isinstance(test, list):
            return Fail(test)
        if len(test) == 0:
            return Fail(test)
        results = [self.rule.apply(child) for child in test]
        # Were there any successes?
        if not any(result.is_success() for result in results):
            return Fail(test)
        # Splice in any list edits.
        new_values = list(
            _expand_edits(
                result.expect_success() if result.is_success() else result.test
                for result in results
            )
        )
        # Were there any successes that returned a different value?
        if _list_unchanged(new_values, test):
            return Success(test, test)
        # Everything succeeded but there was at least one different value.
        return Success(test, new_values)

    def __str__(self) -> str:
        return f"some_list_members( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return False


class SomeChildren(Rule):
    """Apply a rule to all children.  Succeeds if the rule succeeds for one or
    more children, and returns a constructed object with the children replaced
    with the new values. Otherwise, fails."""

    get_children: Callable[[Any], Dict[str, Any]]
    construct: Callable[[type, Dict[str, Any]], Any]
    rule: Rule

    def __init__(
        self,
        rule: Rule,
        get_children: Callable[[Any], Dict[str, Any]],
        construct: Callable[[type, Dict[str, Any]], Any],
        name: str = "some_children",
    ) -> None:
        Rule.__init__(self, name)
        self.rule = rule
        self.get_children = get_children
        self.construct = construct

    def apply(self, test: Any) -> RuleResult:
        children = self.get_children(test)
        # Easy out for leaves.
        if len(children) == 0:
            return Fail(test)
        results = {
            child_name: self.rule.apply(child_value)
            for child_name, child_value in children.items()
        }
        # Were there any successes?
        if not any(result.is_success() for result in results.values()):
            return Fail(test)
        # Were there any successes that returned a different value?
        new_values = {
            n: results[n].expect_success()
            if results[n].is_success()
            else results[n].test
            for n in results
        }
        if all(new_values[n] is children[n] for n in new_values):
            # Everything succeeded and there were no changes.
            return Success(test, test)
        # Everything succeeded but there was at least one different value.
        # Construct a new object.
        return Success(test, self.construct(type(test), new_values))

    def __str__(self) -> str:
        return f"some_children( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        # Fails if there are no children
        return False


class OneListMember(Rule):
    """Apply a rule to all members until the first success.  Succeeds if it
    finds one success and returns a list with the child replaced
    with the new value. Otherwise, fails."""

    rule: Rule

    def __init__(self, rule: Rule, name: str = "one_child") -> None:
        Rule.__init__(self, name)
        self.rule = rule

    def apply(self, test: Any) -> RuleResult:
        if not isinstance(test, list):
            return Fail(test)
        for i, child in enumerate(test):
            result = self.rule.apply(child)
            if result.is_success():
                new_value = result.expect_success()
                if new_value is child:
                    return Success(test, test)
                new_values = test.copy()
                new_values[i] = new_value
                new_values = list(_expand_edits(new_values))
                if _list_unchanged(new_values, test):
                    return Success(test, test)
                return Success(test, new_values)
        return Fail(test)

    def __str__(self) -> str:
        return f"one_list_member( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return False


class OneChild(Rule):
    """Apply a rule to all children until the first success.  Succeeds if it
    finds one success and returns a constructed object with the child replaced
    with the new value. Otherwise, fails."""

    get_children: Callable[[Any], Dict[str, Any]]
    construct: Callable[[type, Dict[str, Any]], Any]
    rule: Rule

    def __init__(
        self,
        rule: Rule,
        get_children: Callable[[Any], Dict[str, Any]],
        construct: Callable[[type, Dict[str, Any]], Any],
        name: str = "one_child",
    ) -> None:
        Rule.__init__(self, name)
        self.rule = rule
        self.get_children = get_children
        self.construct = construct

    def apply(self, test: Any) -> RuleResult:
        children = self.get_children(test)
        for child_name, child_value in children.items():
            result = self.rule.apply(child_value)
            if result.is_success():
                new_value = result.expect_success()
                if new_value is child_value:
                    return Success(test, test)
                new_values = children.copy()
                new_values[child_name] = child_value
                return Success(test, self.construct(type(test), new_values))
        return Fail(test)

    def __str__(self) -> str:
        return f"one_child( {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return False


class SpecificChild(Rule):
    """Apply a rule to a specific child.  If it succeeds, replace the child
    with the new value; otherwise, fail. The child is required to exist."""

    get_children: Callable[[Any], Dict[str, Any]]
    construct: Callable[[type, Dict[str, Any]], Any]
    child: str
    rule: Rule

    def __init__(
        self,
        child: str,
        rule: Rule,
        get_children: Callable[[Any], Dict[str, Any]],
        construct: Callable[[type, Dict[str, Any]], Any],
        name: str = "specific_child",
    ) -> None:
        Rule.__init__(self, name)
        self.rule = rule
        self.get_children = get_children
        self.construct = construct
        self.child = child

    def apply(self, test: Any) -> RuleResult:
        children = self.get_children(test)
        assert self.child in children
        value = children[self.child]
        result = self.rule.apply(value)
        if result.is_fail():
            return Fail(test)
        new_value = result.expect_success()
        if new_value is value:
            return Success(test, test)
        new_values = children.copy()
        new_values[self.child] = new_value
        return Success(test, self.construct(type(test), new_values))

    def __str__(self) -> str:
        return f"specific_child( {self.child}, {str(self.rule)} )"

    def always_succeeds(self) -> bool:
        return self.rule.always_succeeds()


class RuleDomain:
    get_children: Callable[[Any], Dict[str, Any]]
    construct: Callable[[type, Dict[str, Any]], Any]

    def __init__(
        self,
        get_children: Callable[[Any], Dict[str, Any]],
        construct: Callable[[type, Dict[str, Any]], Any],
    ) -> None:
        self.get_children = get_children
        self.construct = construct

    def all_children(self, rule: Rule, name: str = "all_children") -> Rule:
        return AllChildren(rule, self.get_children, self.construct, name)

    def some_children(self, rule: Rule, name: str = "some_children") -> Rule:
        return if_then(
            is_list,
            SomeListMembers(rule),
            SomeChildren(rule, self.get_children, self.construct, name),
        )

    def one_child(self, rule: Rule, name: str = "one_child") -> Rule:
        return if_then(
            is_list,
            OneListMember(rule),
            OneChild(rule, self.get_children, self.construct, name),
        )

    def specific_child(
        self, child: str, rule: Rule, name: str = "specific_child"
    ) -> Rule:
        """Apply a rule to a specific child.  If it succeeds, replace the child
        with the new value; otherwise, fail. The child is required to exist."""
        return SpecificChild(child, rule, self.get_children, self.construct, name)

    # CONSIDER: Should we implement a class for bottom-up traversal, so that
    # CONSIDER: there is a place to put a breakpoint, and so on?
    def bottom_up(self, rule: Rule, name: str = "bottom_up") -> Rule:
        """The bottom-up combinator applies a rule to all leaves, then to the rewritten
        parent, and so on up to the root."""
        return Compose(
            self.all_children(Recursive(lambda: self.bottom_up(rule, name))), rule
        )

    # CONSIDER: Similarly.
    def top_down(self, rule: Rule, name: str = "top_down") -> Rule:
        """The top-down combinator applies a rule to the root, then to the new root's
        children, and so on down to the leaves. It succeeds iff the rule succeeds on
        every node."""
        return Compose(
            rule, self.all_children(Recursive(lambda: self.top_down(rule, name))), name
        )

    def some_top_down(self, rule: Rule, name: str = "some_top_down") -> Rule:
        """The some-top-down combinator is like top_down, in that it applies a
        rule to every node in the tree starting from the top. However, top_down
        requires that the rule succeed for all nodes in the tree; some_top_down
        applies the rule to as many nodes in the tree as possible, leaves alone
        nodes for which it fails (aside from possibly rewriting the children),
        and fails only if the rule applied to no node."""

        # This combinator is particularly useful because it ensures that
        # either progress is made, or the rule fails, and it makes as much
        # progress as possible on each attempt.
        #
        # Note that many(some_top_down(rule)) is a fixpoint combinator.

        return either_or_both(
            rule,
            self.some_children(Recursive(lambda: self.some_top_down(rule, name)), name),
        )

    def some_bottom_up(self, rule: Rule, name: str = "some_bottom_up") -> Rule:
        """The some-bottom-up combinator is like bottom_up, in that it applies a
        rule to every node in the tree starting from the leaves. However, bottom_up
        requires that the rule succeed for all nodes in the tree; some_bottom_up
        applies the rule to as many nodes in the tree as possible, leaves alone
        nodes for which it fails, and fails only if the rule applied to no node."""

        # This combinator is particularly useful because it ensures that
        # either progress is made, or the rule fails, and it makes as much
        # progress as possible on each attempt.
        #
        # Note that many(some_bottom_up(rule)) is a fixpoint combinator.

        return either_or_both(
            self.some_children(
                Recursive(lambda: self.some_bottom_up(rule, name)), name
            ),
            rule,
        )

    # CONSIDER: Similarly.
    def down_then_up(
        self, pre_rule: Rule, post_rule: Rule, name: str = "down_then_up"
    ) -> Rule:
        """The down-then-up combinator is a combination of the bottom-up and
        top-down combinators; it applies the 'pre' rule in a top-down traversal
        and then the 'post' rule on the way back up."""

        return Compose(
            Compose(
                pre_rule,
                self.all_children(
                    Recursive(lambda: self.down_then_up(pre_rule, post_rule, name))
                ),
            ),
            post_rule,
        )

    def descend_until(
        self, test: Rule, rule: Rule, name: str = "descend_until"
    ) -> Rule:
        """descend_until starts at the top of the tree and descends down it
        checking every node to see if "test" succeeds. If it does, it stops
        descending and runs "rule" on that node. It does this on every node
        that meets the test starting from the root."""
        return if_then(
            test,
            rule,
            self.all_children(Recursive(lambda: self.descend_until(test, rule))),
        )
