# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Optional, Union

from beanmachine.ppl.legacy.world.diff import Diff
from beanmachine.ppl.legacy.world.variable import Variable
from beanmachine.ppl.legacy.world.world_vars import WorldVars
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class DiffStack(object):
    """
    Represents the collection of random variables in World based inference.
    """

    def __init__(self, diff: Optional[Diff] = None):
        # diff_stack_ is the list of diffs where the last element is the latest
        # diff added to diff_stack_.
        self.diff_stack_ = []

        # diff_var_ is the set of random variable available in diff stack.
        self.diff_var_ = set()

        # node_to_diffs_ is the dictionary to map node's to the index of the
        # diffs in the diff stack that holds them. This makes node look-up much
        # faster in the diff stack because we no longer need to go look for them
        # in each diff in the diff stack.
        self.node_to_diffs_ = defaultdict(list)

        if diff is None:
            diff = Diff()
        self.add_diff(Diff())

        # diff_ is the latest diff in the diff stack.
        self.diff_ = self.diff_stack_[-1]

    def add_diff(self, diff: Diff) -> None:
        """
        Adds diff to the diff stack.

        :param diff: add diff to the diff stack.
        """
        self.diff_stack_.append(diff)
        self.diff_ = self.diff_stack_[-1]

    def get_node_earlier_version(self, node: RVIdentifier) -> Optional[Variable]:
        """
        Get the earlier version of the node in the diff stack.

        :param node: the node to be looked up in the diff stack.
        :returns: the earlier version of the node in the diff stack if None, it
        means the node is not available in diff stack and instead it can be
        available in the world vars.
        """
        node_indices = self.node_to_diffs_[node]
        if not node_indices:
            raise ValueError(f"Node {node} is missing")
        if len(node_indices) == 1:
            return None

        return self.diff_stack_[node_indices[-2]].get_node_raise_error(node)

    def add_node(self, node: RVIdentifier, value: Variable) -> None:
        """
        Add/update the node variable in the diff.
        """
        self.diff_stack_[-1].add_node(node, value)
        if len(self.node_to_diffs_[node]) == 0 or self.node_to_diffs_[node][-1] != (
            len(self.diff_stack_) - 1
        ):
            self.diff_var_.add(node)
            self.node_to_diffs_[node].append(self.len() - 1)

    def len(self) -> int:
        """
        :returns: the length of the diff stack
        """
        return len(self.diff_stack_)

    def remove_last_diff(self) -> Diff:
        """
        Delete latest diff and returns the new latest diff

        :returns: the new latest diff
        """
        diff_len = self.len() - 1
        for node in self.diff_.vars():
            node_indices = self.node_to_diffs_[node]
            if diff_len in node_indices:
                node_indices.remove(diff_len)

        self.diff_ = self.diff_stack_[-2]
        del self.diff_stack_[-1]
        return self.diff_

    def get_diff_stack(self) -> List:
        """
        :return: the diff stack
        """
        return self.diff_stack_

    def reset(self) -> None:
        """
        Resets the diff stacks.
        """
        self.diff_stack_ = []

    def top(self) -> Diff:
        """
        Get the latest diff on the diff stack.

        :returns: the latest diff on the stack.
        """
        return self.diff_stack_[-1]

    def is_marked_for_delete(self, node: RVIdentifier) -> bool:
        """
        Get whether a node is marked for delete.

        :param node: the RVIdentifier to be looked up in the diff stack.
        :returns: whether a node is marked for delete.
        """
        if node in self.node_to_diffs_ and self.node_to_diffs_[node]:
            diff_index = self.node_to_diffs_[node][-1]
            if self.diff_stack_[diff_index].is_marked_for_delete(node):
                return True

        return False

    def get_node(self, node: RVIdentifier) -> Optional[Variable]:
        """
        Get the node from diff stack.

        :param node: the RVIdentifier to be looked up in the diff stack.
        :returns: the latest node Variable available in diff stack and returns
        None if not available.
        """
        if node in self.node_to_diffs_ and self.node_to_diffs_[node]:
            diff_index = self.node_to_diffs_[node][-1]
            if not self.diff_stack_[diff_index].is_marked_for_delete(node):
                return self.diff_stack_[diff_index].get_node(node)

        return None

    def get_node_raise_error(self, node: RVIdentifier) -> Variable:
        """
        Get the node from diff stack and raise an error if not available.

        :param node: the RVIdentifier to be looked up in the diff stack.
        :returns: the latest node Variable available in diff stack.
        """
        node_var = self.get_node(node)
        if node_var is not None:
            return node_var

        raise ValueError(f"Node {node} is missing")

    def contains_node(self, node: RVIdentifier) -> bool:
        """
        :returns: whether the diff stack contains the node.
        """
        return node in self.node_to_diffs_ and len(self.node_to_diffs_[node]) != 0

    def update_log_prob(self, value: Tensor) -> None:
        """
        Update the log prob of the latest diff in the diff stack.

        :param value: add value to the log prob of the latest diff in the diff
        stack.
        """
        self.diff_.update_log_prob(value.detach())

    def mark_for_delete(self, node: RVIdentifier) -> None:
        """
        Marks the node to be deleted in the latest diff in the diff stack.

        :param node: the node to be marked to be deleted in the diff.
        """
        self.diff_.mark_for_delete(node)

    def diffs_log_prob(self) -> Union[float, Tensor]:
        """
        :returns: the log_prob of all diffs in the diff stack.
        """
        full_log_prob = 0.0
        for x in self.diff_stack_:
            full_log_prob += x.log_prob()
        return full_log_prob

    def push_changes(self, world_vars: WorldVars) -> None:
        """
        Push changes of the diff_stack over to world_vars.

        :param world_vars: the world variable to push the diff stack changes to.
        """
        for node in self.diff_var_:
            node_indices = self.node_to_diffs_[node]
            if node_indices:
                is_marked_for_delete = self.diff_stack_[
                    node_indices[-1]
                ].is_marked_for_delete(node)
                if is_marked_for_delete:
                    world_vars.delete(node)
                else:
                    node_var = self.diff_stack_[node_indices[-1]].get_node(node)
                    world_vars.add_node(node, node_var)
