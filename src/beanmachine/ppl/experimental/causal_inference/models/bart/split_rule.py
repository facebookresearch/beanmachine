# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from dataclasses import dataclass
from typing import List, Optional

import torch


class Operator(enum.Enum):
    le = "less than equal to"
    gt = "greater than"


@dataclass(eq=True)
class SplitRule:
    """
    A representation of a split in feature space as a result of a decision node node growing to a leaf node.

    Args:
        grow_dim: The dimension used for the split.
        grow_val: The value used for splitting.
        operator: The relational operation used for the split. The two operators considered are
            "less than or equal" for the left child and "greater than" for the right child.

    """

    __slots__ = ["grow_dim", "grow_val", "operator"]

    def __init__(
        self,
        grow_dim: int,
        grow_val: float,
        operator: Operator,
    ):
        self.grow_dim = grow_dim
        self.grow_val = grow_val
        self.operator = operator


class DimensionalRule:
    """
    Represents the range of values along one dimension of the input which passes a rule.
    For example, if input is X = [x1, x2] then a dimensional rule for x1 could be
    x1 in [3, 4 , 5...20] representing the rule 3 < x1 <=20 (assuming x1 is an integer).

    Args:
        grow_dim: The dimension used for the rule.
        min_val: The minimum value of grow_dim which satisfies the rule (exclusive i.e. min_val fails the rule).
        max_val: The maximum value of grow_dim which satisfies the rule (inclusive i.e. max_val passes the rule).
    """

    def __init__(self, grow_dim: int, min_val: float, max_val: float):
        self.grow_dim = grow_dim
        self.min_val, self.max_val = min_val, max_val

    def add_rule(self, new_rule: SplitRule) -> "DimensionalRule":
        """Add a rule to the dimension. If the rule is less restrictive than an existing rule, nothing changes.

        Args:
            new_rule: The new rule to add.
        """
        if self.grow_dim != new_rule.grow_dim:
            raise ValueError("New rule grow dimension does not match")
        if new_rule.operator == Operator.gt and new_rule.grow_val > self.min_val:
            return DimensionalRule(self.grow_dim, new_rule.grow_val, self.max_val)
        elif new_rule.operator == Operator.le and new_rule.grow_val < self.max_val:
            return DimensionalRule(self.grow_dim, self.min_val, new_rule.grow_val)
        else:
            # new rule is already covered by existing rule
            return self


class CompositeRules:
    """
    Represents a composition of `DimensionalRule`s along multiple dimensions of input.
    For example, if input is X = [x1, x2] then a composite rule could be
    x1 in [3, 4 , 5...20] and x2 in [-inf..-10] representing the rule 3 < x1 <=20
    (assuming x1 is an integer) and x2<= -10.

    Note:
        CompositeRules arre immutable and all changes to them return copies with the desired modification.

    Args:
        all_dims: All dimensions which have rules.
        all_split_rules: All rules corresponding to each dimension in `all_dims`.
    """

    def __init__(
        self, all_dims: List[int], all_split_rules: Optional[List[SplitRule]] = None
    ):
        self.dimensional_rules = {
            dim: DimensionalRule(dim, -float("inf"), float("inf")) for dim in all_dims
        }
        if all_split_rules is None:
            self.all_split_rules = []
        else:
            self.all_split_rules = all_split_rules

        for split_rule in self.all_split_rules:
            self.dimensional_rules[split_rule.grow_dim] = self.dimensional_rules[
                split_rule.grow_dim
            ].add_rule(split_rule)
        if len(self.all_split_rules) > 0:
            self.grow_dim = self.all_split_rules[-1].grow_dim
        else:
            self.grow_dim = None

    def condition_on_rules(self, X: torch.Tensor) -> torch.Tensor:
        """Condition the input on a composite rule and get a mask such that X[mask]
        satisfies the rule.

        Args:
            X: Input / covariate matrix.
        """
        mask = torch.ones(len(X), dtype=torch.bool)
        for dim in self.dimensional_rules.keys():
            mask = (
                mask
                & (X[:, dim].gt(self.dimensional_rules[dim].min_val))
                & (X[:, dim].le(self.dimensional_rules[dim].max_val))
            )
        return mask

    def add_rule(self, new_rule: SplitRule) -> "CompositeRules":
        """Add a split rule to the composite ruleset. Returns a copy of `CompositeRules`"""
        if new_rule.grow_dim not in self.dimensional_rules.keys():
            raise ValueError(
                "The dimension of new split rule is outside the scope of the composite rule"
            )

        return CompositeRules(
            list(self.dimensional_rules.keys()), self.all_split_rules + [new_rule]
        )

    def most_recent_split_rule(self) -> Optional[SplitRule]:
        """Returns the most recent split_rule added. Returns None if no rules were applied."""
        if len(self.all_split_rules) == 0:
            return None
        else:
            return self.all_split_rules[-1]
