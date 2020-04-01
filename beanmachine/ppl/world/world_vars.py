# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from typing import Dict, Optional

from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world.variable import Variable
from torch import Tensor, tensor


class WorldVars(object):
    """
    Represents the collection of random variables in Word based inference.
    """

    def __init__(self):
        self.data_ = defaultdict()
        self.log_prob_ = tensor(0.0)

    def add_node(self, node: RVIdentifier, value: Variable) -> None:
        """
        Add and/or update the node to the WorldVars with its corresponding
        random variables.

        :params node: the RVIdentifier (key) of the node to be added/updated in
        the WorldVars.
        :params value: the Variable corresponding to the node to be added/updated
        in the WorldVars.
        """
        self.data_[node] = value

    def get_node(self, node: RVIdentifier) -> Optional[Variable]:
        """
        Get the node in WorldVars.

        :params node: the RVIdentifier of the node to be looked up in WorldVars.
        :returns: the variable of the node in WorldVars and None if the node
        does not exist.
        """
        if node in self.data_:
            return self.data_[node]
        return None

    def get_node_raise_error(self, node: RVIdentifier) -> Variable:
        """
        Get the node in WorldVars and raise a ValueError if the node does not
        exists.

        :params node: the RVIdentifier of node to be looked up in WorldVars.
        :returns: the variable of the node in WorldVars and raises an error if
        the node does not exist.
        """
        var = self.get_node(node)
        if var is None:
            raise ValueError(f"Node {node} is missing")
        return var

    def contains_node(self, node: RVIdentifier) -> bool:
        """
        Returns whether the WorlVars contains the node.

        :params node: the RVIdentifier (key) of the node to looked up in the
        worldvars.
        :returns: whether the node exists in the diff.
        """
        return node in self.data_

    def delete(self, node) -> None:
        """
        Delete node's random variable.

        :param node: the node to delete
        """
        del self.data_[node]

    def len(self) -> int:
        """
        :returns: the number of random variable in WorldVars.
        """
        return len(self.data_)

    def vars(self) -> Dict:
        """
        :returns: the collection of random variables.
        """
        return self.data_

    def update_log_prob(self, value: Tensor) -> None:
        """
        Update the log_prob with value.

        :params value: the value to update the log_prob with.
        """
        self.log_prob_ += value
