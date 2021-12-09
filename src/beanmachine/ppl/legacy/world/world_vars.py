# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Optional, Union

from beanmachine.ppl.legacy.world.variable import Variable
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class WorldVars(object):
    """
    Represents the collection of random variables in World based inference.
    """

    log_prob_: Union[float, Tensor]

    def __init__(self):
        self.var_funcs_ = defaultdict(set)
        self.data_ = defaultdict()
        self.log_prob_ = 0.0

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
        if hasattr(node, "function"):
            self.var_funcs_[node.wrapper].add(node)

    def contains_node_by_func(self, node_func: str) -> bool:
        """
        Returns whether the WorlVars contains the node with function node_func.

        :params node_func: the node_func of the node to looked up in the
        worldvars.
        :returns: whether a node with node_func exists in WorldVars.
        """
        if node_func in self.var_funcs_:
            return True
        return False

    def get_nodes_by_func(self, node_func: str):
        """
        Get the nodes in WorldVars with the given node_func.

        :params node_func: the node_func to be looked up in WorldVars.
        :returns: the variable of the node in WorldVars and None if the node
        does not exist.
        """
        return self.var_funcs_[node_func]

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
        :returns: whether the node exists in WorldVars.
        """
        return node in self.data_

    def delete(self, node) -> None:
        """
        Delete node's random variable.

        :param node: the node to delete
        """
        del self.data_[node]
        self.var_funcs_[node.wrapper].remove(node)

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
