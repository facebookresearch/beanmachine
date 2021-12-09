#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A pattern matching engine"""
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union


# Logically, a pattern is just a predicate; it's a function from
# any value to bool: True if the value matches the predicate,
# false otherwise.
#
# However it will be convenient for us to be able to represent
# patterns like "is equal to zero" as just the integer 0,
# "is an instance of type T" as just the type object.
#
# Similarly, it will be convenient to build complex patterns as
# compositions of simpler ones.
#
# Also, for debugging purposes we will not simply return True
# or False; we'll return a MatchResult object that indicates
# what value was matched, whether it was successfully matched
# or not, and, if the pattern had subpatterns, what the results
# of those were.


# TODO: Tensor comprehension patterns


_empty = {}


class MatchResult(ABC):
    """The result of a pattern match; either success or failure."""

    test: Any
    submatches: Dict[str, "MatchResult"]

    def __init__(
        self, test: Any, submatches: Optional[Dict[str, "MatchResult"]] = None
    ) -> None:
        self.test = test
        self.submatches = submatches if submatches is not None else _empty

    @abstractmethod
    def is_success(self) -> bool:
        pass

    @abstractmethod
    def is_fail(self) -> bool:
        pass

    def __str__(self) -> str:
        return f"{type(self).__name__}:{self.test}"

    def __bool__(self) -> bool:
        return self.is_success()

    # TODO: Display as a tree and graph also


class Fail(MatchResult):
    """A pattern that always fails."""

    # TODO: If we save the patterns that failed as well, then we can build a
    # TODO: diagnostic engine that describes why a value failed to match against
    # TODO: a complex pattern.
    def __init__(
        self, test: Any = None, submatches: Optional[Dict[str, MatchResult]] = None
    ) -> None:
        MatchResult.__init__(self, test, submatches)

    def is_success(self) -> bool:
        return False

    def is_fail(self) -> bool:
        return True


class Success(MatchResult):
    """A pattern that always succeeds."""

    def __init__(
        self, test: Any = None, submatches: Optional[Dict[str, MatchResult]] = None
    ) -> None:
        MatchResult.__init__(self, test, submatches)

    def is_success(self) -> bool:
        return True

    def is_fail(self) -> bool:
        return False


class PatternBase(ABC):
    @abstractmethod
    def match(self, test: Any) -> MatchResult:
        pass

    @abstractmethod
    def _to_str(self, test: str) -> str:
        pass

    def __str__(self) -> str:
        return self._to_str("test")

    def __call__(self, test: Any) -> MatchResult:
        return self.match(test)


Pattern = Union[PatternBase, int, str, float, type, list, None]


class PredicatePattern(PatternBase):
    """A pattern is logically a predicate; this pattern just encapsulates any
    predicate that returns a match result."""

    predicate: Callable[[Any], bool]
    name: str

    def __init__(self, predicate: Callable[[Any], bool], name: str = "if") -> None:
        self.predicate = predicate
        self.name = name

    def match(self, test: Any) -> MatchResult:
        return Success(test) if self.predicate(test) else Fail(test)

    def _to_str(self, test: str) -> str:
        return f"{self.name}({test})"


AtomicType = Union[bool, int, float, str, None]


class AtomicPattern(PatternBase, metaclass=ABCMeta):
    """An atomic pattern matches against a single specific value, such as a
    specific integer, Boolean, string, and so on."""

    value: AtomicType

    def __init__(self, value: AtomicType) -> None:
        self.value = value

    def match(self, test: Any) -> MatchResult:
        return Success(test) if test == self.value else Fail(test)

    def _to_str(self, test: str) -> str:
        return f"{test}=={str(self.value)}"


class BoolPattern(AtomicPattern):
    """The pattern that matches a specific Boolean value."""

    def __init__(self, value: bool) -> None:
        AtomicPattern.__init__(self, value)


truePattern = BoolPattern(True)
falsePattern = BoolPattern(False)


class IntPattern(AtomicPattern):
    """The pattern that matches a specific integer value."""

    def __init__(self, value: int) -> None:
        AtomicPattern.__init__(self, value)


class FloatPattern(AtomicPattern):
    """The pattern that matches a specific float value."""

    def __init__(self, value: float) -> None:
        AtomicPattern.__init__(self, value)


class StringPattern(AtomicPattern):
    """The pattern that matches a specific string value."""

    def __init__(self, value: str) -> None:
        AtomicPattern.__init__(self, value)


# Note that we do not want to use "None" to mean "the pattern that matches nothing"
# because it will be useful to be able to match "None" values in ASTs. Use the
# FailPattern if you want a pattern that never matches.
class NonePattern(AtomicPattern):
    """The pattern that matches None."""

    def __init__(self) -> None:
        AtomicPattern.__init__(self, None)

    def _to_str(self, test: str) -> str:
        return f"{test} is None"


nonePattern = NonePattern()


class AnyPattern(PatternBase):
    """The pattern that matches anything; it always succeeds."""

    def __init__(self) -> None:
        pass

    def match(self, test: Any) -> MatchResult:
        return Success(test)

    def _to_str(self, test: str) -> str:
        return f"{test} is Any"


anyPattern = AnyPattern()


def is_any(pattern: Pattern) -> bool:
    return isinstance(pattern, AnyPattern)


class FailPattern(PatternBase):
    """The pattern that matches nothing; it always fails."""

    def __init__(self) -> None:
        pass

    def match(self, test: Any) -> MatchResult:
        return Fail(test)

    def _to_str(self, test: str) -> str:
        return "FAIL"


failPattern = FailPattern()


class TypePattern(PatternBase):
    """The pattern that matches if the value is an instance of the given type."""

    typ: type

    def __init__(self, typ: type) -> None:
        self.typ = typ

    def match(self, test: Any) -> MatchResult:
        return Success(test) if isinstance(test, self.typ) else Fail(test)

    def _to_str(self, test: str) -> str:
        return f"isinstance({test}, {self.typ.__name__})"


def _match_list_pattern(patterns: List[Pattern], test: Any) -> MatchResult:
    if not isinstance(test, list) or len(test) != len(patterns):
        return Fail(test)
    submatches = {str(i): match(pattern, test[i]) for i, pattern in enumerate(patterns)}
    if any(result.is_fail() for result in submatches.values()):
        return Fail(test, submatches)
    return Success(test, submatches)


def match(pattern: Pattern, test: Any) -> MatchResult:
    if pattern is None:
        return Success(test) if test is None else Fail(test)
    if (
        isinstance(pattern, int)
        or isinstance(pattern, str)
        or isinstance(pattern, bool)
        or isinstance(pattern, float)
    ):
        return Success(test) if test == pattern else Fail(test)
    if isinstance(pattern, list):
        return _match_list_pattern(pattern, test)
    if isinstance(pattern, type):
        return Success(test) if isinstance(test, pattern) else Fail(test)
    if isinstance(pattern, PatternBase):
        return pattern.match(test)
    raise TypeError(f"Expected pattern, got {type(pattern).__name__}")


def to_pattern(pattern: Pattern) -> PatternBase:
    """Takes any value that can be used as a pattern, and returns an object that
    derives from PatternBase that has the same semantics."""
    if isinstance(pattern, PatternBase):
        return pattern
    if pattern is None:
        return nonePattern
    if isinstance(pattern, bool):
        return BoolPattern(pattern)
    if isinstance(pattern, int):
        return IntPattern(pattern)
    if isinstance(pattern, float):
        return FloatPattern(pattern)
    if isinstance(pattern, str):
        return StringPattern(pattern)
    if isinstance(pattern, list):
        if len(pattern) == 0:
            return EmptyListPattern()
        return ListPattern(pattern)
    if isinstance(pattern, type):
        return TypePattern(pattern)
    raise TypeError(f"Expected pattern, got {type(pattern).__name__}")


class Negate(PatternBase):
    """Negates a pattern; if the underlying pattern succeeds, this fails, and
    vice versa."""

    pattern: Pattern

    def __init__(self, pattern: Pattern) -> None:
        self.pattern = pattern

    def match(self, test: Any) -> MatchResult:
        result = match(self.pattern, test)
        if result.is_success():
            return Fail(result.test, result.submatches)
        return Success(result.test, result.submatches)

    def _to_str(self, test: str) -> str:
        return f"not({to_pattern(self.pattern)._to_str(test)})"


# This is a *pattern combinator*. It takes a pattern and returns a modification
# of the pattern; in this case, it's negation.
def negate(pattern: Pattern) -> Pattern:
    """Produces the negation of a given pattern."""
    if isinstance(pattern, Negate):
        return pattern.pattern
    return Negate(pattern)


class MatchEvery(PatternBase):
    """The pattern that succeeds if every pattern in the list succeeds.
    It will stop trying to match patterns after the first failure. If there
    are no patterns in the list then it succeeds."""

    patterns: List[Pattern]

    def __init__(self, *patterns: Pattern) -> None:
        self.patterns = list(patterns)

    def match(self, test: Any) -> MatchResult:
        submatches = {}
        for p in self.patterns:
            result = match(p, test)
            submatches.update(result.submatches)
            if result.is_fail():
                # We return the submatches here just for diagnostic purposes; since
                # this pattern intends to match *every* subpattern, it is helpful
                # when diagnosing a failure to see what all the failed submatches were.
                return Fail(test, submatches)
        return Success(test, submatches)

    def _to_str(self, test: str) -> str:
        children = " and ".join(
            to_pattern(pattern)._to_str(test) for pattern in self.patterns
        )
        return f"({children})"


# This is also a pattern combinator.
def match_every(*patterns: Pattern) -> Pattern:
    ps: List[Pattern] = list(patterns)
    while True:
        # If there is an "any" in the list we can discard it.
        ps = [p for p in ps if not is_any(p)]
        if len(ps) == 0:
            return anyPattern
        if len(ps) == 1:
            return ps[0]
        if any(p is FailPattern for p in ps):
            return failPattern
        index = next(
            (i for (i, pattern) in enumerate(ps) if isinstance(pattern, MatchEvery)),
            None,
        )
        if index is None:
            return MatchEvery(*ps)
        child = ps[index]
        assert isinstance(child, MatchEvery)
        ps = ps[:index] + child.patterns + ps[(index + 1) :]


class MatchAny(PatternBase):
    """The pattern that succeeds if any pattern in the list succeeds.
    It will stop trying to match patterns after the first success. If there
    are no patterns in the list then it fails."""

    patterns: List[Pattern]

    def __init__(self, *patterns: Pattern) -> None:
        self.patterns = list(patterns)

    def match(self, test: Any) -> MatchResult:
        # Bail out on the first success.
        submatches = {}
        for p in self.patterns:
            result = match(p, test)
            submatches.update(result.submatches)
            if result.is_success():
                return result
        return Fail(test, submatches)

    def _to_str(self, test: str) -> str:
        children = " or ".join(
            to_pattern(pattern)._to_str(test) for pattern in self.patterns
        )
        return f"({children})"


# Another combinator.
def match_any(*patterns: Pattern) -> Pattern:
    ps: List[Pattern] = list(patterns)
    while True:
        # If there is a "Fail" in the list we can discard it.
        ps = [p for p in ps if not isinstance(p, FailPattern)]
        if len(ps) == 0:
            return failPattern
        if len(ps) == 1:
            return ps[0]
        if any(is_any(p) for p in ps):
            return anyPattern
        index = next(
            (i for (i, pattern) in enumerate(ps) if isinstance(pattern, MatchAny)), None
        )
        if index is None:
            return MatchAny(*ps)
        child = ps[index]
        assert isinstance(child, MatchAny)
        ps = ps[:index] + child.patterns + ps[index + 1 :]


class Subpattern(PatternBase):
    """Sometimes we want to check to see if a given value matches the pattern
    whereby some projected value matches a pattern. This class represents
    such patterns. It takes a subpattern and a projection; when match is called,
    it projects the value and runs the subpattern on the projected value."""

    name: str
    subpattern: Pattern
    get_subtest: Callable[[Any], Any]

    def __init__(
        self, name: str, subpattern: Pattern, get_subtest: Callable[[Any], Any]
    ) -> None:
        self.name = name
        self.subpattern = subpattern
        self.get_subtest = get_subtest

    def match(self, test: Any) -> MatchResult:
        submatch = match(self.subpattern, self.get_subtest(test))
        submatches = {self.name: submatch}
        if submatch.is_success():
            return Success(test, submatches)
        return Fail(test, submatches)

    def _to_str(self, test: str) -> str:
        return to_pattern(self.subpattern)._to_str(f"{test}.{self.name}")


class AttributeSubpattern(PatternBase):
    """Sometimes we want to check to see if an attribute of a value matches
    a pattern. This class represents such patterns. It takes a subpattern and
    an attribute name. When match is called, it runs the subpattern on the
    attribute of the given value."""

    name: str
    subpattern: Pattern

    def __init__(self, name: str, subpattern: Pattern) -> None:
        self.name = name
        self.subpattern = subpattern

    def match(self, test: Any) -> MatchResult:
        submatch = match(self.subpattern, getattr(test, self.name, None))
        submatches = {self.name: submatch}
        if submatch.is_success():
            return Success(test, submatches)
        return Fail(test, submatches)

    def _to_str(self, test: str) -> str:
        return to_pattern(self.subpattern)._to_str(f"{test}.{self.name}")


# Another combinator
def attribute(name: str, subpattern: Pattern) -> Pattern:
    if is_any(subpattern):
        return subpattern
    return AttributeSubpattern(name, subpattern)


class EmptyListPattern(PatternBase):
    """This pattern matches an empty list."""

    name: str

    def __init__(self, name: str = "empty_list") -> None:
        self.name = name

    def match(self, test: Any) -> MatchResult:
        if isinstance(test, list) and len(test) == 0:
            return Success(test)
        return Fail(test)

    def _to_str(self, test: str) -> str:
        return f"{test}==[]"


emptyList = EmptyListPattern()

nonEmptyList = match_every(list, negate(emptyList))

twoPlusList = match_every(
    list, negate(match_any([], [anyPattern], [anyPattern, anyPattern]))
)


class HeadTail(PatternBase):
    """This combinator takes a pattern to match the head of a list and
    a pattern to match the tail. If the list is empty, it automatically
    fails; otherwise both patterns must match.  The tail pattern is not
    attempted if the head pattern fails."""

    name: str
    head: Pattern
    tail: Pattern

    def __init__(
        self,
        head: Pattern = anyPattern,
        tail: Pattern = anyPattern,
        name: str = "head_tail",
    ) -> None:
        self.name = name
        self.head = head
        self.tail = tail

    def match(self, test: Any) -> MatchResult:
        if not isinstance(test, list) or len(test) == 0:
            return Fail(test)
        # Python allows this interesting list destructuring:
        h, *t = test
        head_result = match(self.head, h)
        if head_result.is_fail():
            return Fail(test, {"head": h})
        tail_result = match(self.tail, t)
        submatches = {"head": head_result, "tail": tail_result}
        if tail_result.is_fail():
            return Fail(test, submatches)
        return Success(test, submatches)

    def _to_str(self, test: str) -> str:
        h = to_pattern(self.head)._to_str(test + "[0]")
        t = to_pattern(self.tail)._to_str(test + "[1:]")
        return f"{h} and {t}"


class ListPattern(PatternBase):
    """This pattern matches a list of patterns to a list."""

    name: str
    patterns: List[Pattern]

    def __init__(self, patterns: List[Pattern], name: str = "list_pattern") -> None:
        self.patterns = patterns
        self.name = name

    def match(self, test: Any) -> MatchResult:
        return _match_list_pattern(self.patterns, test)

    def _to_str(self, test: str) -> str:
        ps = ", ".join(
            to_pattern(p)._to_str(f"{test}[{i}]") for i, p in enumerate(self.patterns)
        )
        return f"[{ps}]"


class ListAny(PatternBase):
    """Matches a list where one or more elements of the list match the pattern."""

    name: str
    pattern: Pattern

    def __init__(self, pattern: Pattern, name: str = "list_any") -> None:
        self.pattern = pattern
        self.name = name

    def match(self, test: Any) -> MatchResult:
        if not isinstance(test, list):
            return Fail(test)
        # TODO: We could bail out after the first success.
        submatches = {str(i): match(self.pattern, t) for i, t in enumerate(test)}
        if any(result.is_success() for result in submatches.values()):
            return Success(test, submatches)
        return Fail(test, submatches)

    def _to_str(self, test: str) -> str:
        return f"{test}.any(x:{to_pattern(self.pattern)._to_str('x')})"


class ListAll(PatternBase):
    """Matches a list where all elements of the list match the pattern."""

    name: str
    pattern: Pattern

    def __init__(self, pattern: Pattern, name: str = "list_all") -> None:
        self.pattern = pattern
        self.name = name

    def match(self, test: Any) -> MatchResult:
        if not isinstance(test, list):
            return Fail(test)
        # TODO: We could bail out after the first failure.
        submatches = {str(i): match(self.pattern, t) for i, t in enumerate(test)}
        if any(result.is_fail() for result in submatches.values()):
            return Fail(test, submatches)
        return Success(test, submatches)

    def _to_str(self, test: str) -> str:
        return f"{test}.all(x:{to_pattern(self.pattern)._to_str('x')})"


# This complex combinator takes a type and a list of patterns to match against
# attributes of the type, and then returns the resulting pattern that matches
# values of that type with attributes that match all the given patterns.
def type_and_attributes(typ: type, patterns: Dict[str, Pattern]) -> Pattern:
    t: List[Pattern] = [typ]
    tuples: List[Pattern] = [
        attribute(name, subpattern) for (name, subpattern) in patterns.items()
    ]
    return match_every(*(t + tuples))
