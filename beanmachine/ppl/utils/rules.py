#!/usr/bin/env python3
"""A rules engine for tree transformation"""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple

from beanmachine.ppl.utils.patterns import (
    MatchResult,
    Pattern,
    anyPattern,
    failPattern,
    match,
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
    rule = fail
    for pattern, action in pairs:
        rule = or_else(PatternRule(pattern, action, name), rule, name)
    return rule


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


def compose(first: Rule, second: Rule, name: str = "compose") -> Rule:
    """Apply the first rule; if it fails, the operation fails. If it succeeds,
    apply the second rule to its output.
    That is, Compose(a, b)(test) has the semantics of b(a(test))"""
    return Choose(first, second, fail, name)


class Recursive(Rule):
    """Delay construction of a rule until we need it, so as to avoid recursion."""

    rule_maker: Callable[[], Rule]

    def __init__(self, rule_maker: Callable[[], Rule], name: str = "recursive") -> None:
        Rule.__init__(self, name)
        self.rule_maker = rule_maker

    def apply(self, test: Any) -> RuleResult:
        return self.rule_maker().apply(test)


def or_else(first: Rule, second: Rule, name: str = "or_else") -> Rule:
    """Try the first rule. If it succeeds, pass its result to identity, which
    always succeeds. If it fails, try the second rule."""
    return Choose(first, identity, second, name)


def try_once(rule: Rule, name: str = "try_once") -> Rule:
    """Try the rule; if it succeeds, use the result. If it fails use the test
    value."""
    return or_else(rule, identity, name)


def try_many(rule: Rule, name: str = "try_many") -> Rule:
    """Try the rule; if it succeeds, try it again with the result, and so on.
    When it eventually fails, return the final result."""
    return try_once(compose(rule, Recursive(lambda: try_many(rule, name))), name)
