#!/usr/bin/env python3
"""A rules engine for tree transformation"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Tuple

from beanmachine.ppl.utils.patterns import (
    MatchResult,
    Pattern,
    anyPattern,
    failPattern,
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


class Fail(RuleResult):
    def __init__(self, test: Any = None) -> None:
        # pyre-fixme[6]: Expected `MatchResult` for 1st param but got `Fail`.
        MatchResult.__init__(self, test)

    def is_success(self) -> bool:
        return False

    def is_fail(self) -> bool:
        return True

    def expect_success(self) -> Any:
        raise ValueError("Expected success")


class Success(RuleResult):
    result: Any

    def __init__(self, test: Any, result: Any) -> None:
        # pyre-fixme[6]: Expected `MatchResult` for 1st param but got `Success`.
        MatchResult.__init__(self, test)
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


class PatternRule(Rule):
    """If the test value matches the pattern, then the test value is passed
    to the projection and the rule succeeds. Otherwise, the rule fails."""

    pattern: Pattern
    projection: Callable[[Any], Any]

    def __init__(
        self, pattern: Pattern, projection: Callable[[Any], Any], name: str = "pattern"
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


def _identity(x: Any) -> Any:
    return x


# The identity rule is the rule that always succeeds, and the projection
# is an identity function.
identity: Rule = PatternRule(anyPattern, _identity, "identity")
# The fail rule is the rule that never succeeds.
fail: Rule = PatternRule(failPattern, _identity, "fail")


def pattern_rules(
    pairs: List[Tuple[Pattern, Callable[[Any], Any]]], name: str = "pattern_rules"
) -> Rule:
    """Constructs a rule from a sequence of pairs of patterns and projections.
    Patterns are checked in order, and the first one that matches is used for the
    projection; if none match then the rule fails."""
    rules = (PatternRule(pattern, action, name) for pattern, action in pairs)
    return FirstMatch(rules)


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

    def __init__(self, edits: List[Any]):
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


class AllChildren(Rule):
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

    def _apply_to_list(self, test: List[Any]) -> RuleResult:
        # Easy out:
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

    def apply(self, test: Any) -> RuleResult:
        if isinstance(test, list):
            return self._apply_to_list(test)
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
        return f"all_children( {str(self.rule)} )"


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

    def _apply_to_list(self, test: List[Any]) -> RuleResult:
        # Easy out:
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

    def apply(self, test: Any) -> RuleResult:
        if isinstance(test, list):
            return self._apply_to_list(test)
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
        return f"all_children( {str(self.rule)} )"


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

    def _apply_to_list(self, test: List[Any]) -> RuleResult:
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

    def apply(self, test: Any) -> RuleResult:
        if isinstance(test, list):
            return self._apply_to_list(test)
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
        return SomeChildren(rule, self.get_children, self.construct, name)

    def one_child(self, rule: Rule, name: str = "one_child") -> Rule:
        return OneChild(rule, self.get_children, self.construct, name)
