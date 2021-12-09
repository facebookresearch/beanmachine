# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Optional, Union

from beanmachine.ppl.legacy.world.variable import Variable
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class Diff(object):
    """
    Represents the collection of random variables in Word based inference, which
    handles making updates to world random variable in each iteration of
    inference.
    """

    log_prob_: Union[float, Tensor]

    def __init__(self):
        """
        Diff contains data_, collection of random variables, the update to the
        log probability of the world and a dictionary of variables that are
        marked to be deleted.
        """
        self.data_ = defaultdict()
        self.log_prob_ = 0.0
        self.is_delete_ = defaultdict(bool)

    def add_node(self, node: RVIdentifier, value: Variable) -> None:
        """
        Add and/or update the node to the diff with its corresponding random
        variables.

        :params node: the RVIdentifier (key) of the node to be added/updated in
        the diff.
        :params value: the Variable corresponding to the node to be added/updated
        in the diff.
        """
        self.data_[node] = value

    def contains_node(self, node: RVIdentifier) -> bool:
        """
        Returns whether the Diff contains the node.

        :params node: the RVIdentifier (key) of the node to looked up in the diff.
        :returns: whether the node exists in the diff.
        """
        return node in self.data_

    def log_prob(self) -> Union[float, Tensor]:
        """
        :returns: the log probability update of the diff.
        """
        return self.log_prob_

    def update_log_prob(self, value: Tensor) -> None:
        """
        Update the log_prob with value.

        :params value: the value to update the log_prob with.
        """
        self.log_prob_ += value

    def is_marked_for_delete(self, node: RVIdentifier) -> bool:
        """
        Returns whether a node is marked to be deleted in the diff.

        :params node: the RVIdentifier of the node to be looked up.
        :returns: whether the node is marked for delete in the diff.
        """
        return self.is_delete_[node]

    def mark_for_delete(self, node: RVIdentifier):
        self.is_delete_[node] = True

    def get_node(self, node: RVIdentifier) -> Optional[Variable]:
        """
        Get the node in the diff.

        :params node: the RVIdentifier of the node to be looked up in the diff.
        :returns: the variable of the node in the diff and None if the node
        does not exist.
        """
        if node in self.data_:
            return self.data_[node]
        return None

    def get_node_raise_error(self, node: RVIdentifier) -> Variable:
        """
        Get the node in the diff and raise a ValueError if the node does not
        exists in the diff.

        :params node: the RVIdentifier of node to be looked up in the diff.
        :returns: the variable of the node in the diff and raises an error if
        the node does not exist.
        """
        var = self.get_node(node)
        if var is None:
            raise ValueError(f"Node {node} is missing")
        return var

    def vars(self) -> Dict:
        """
        :returns: the collection of random variables.
        """
        return self.data_

    def to_be_deleted_vars(self) -> Dict:
        """
        :returns: a dictionary that includes whether a random variable is marked
        to be deleted or not.
        """
        return self.is_delete_

    def delete(self, node) -> None:
        """
        Delete node's random variable.

        :param node: the node to delete
        """
        del self.data_[node]

    def len(self) -> int:
        """
        :returns: the number of random variable in the diff.
        """
        return len(self.data_)
