# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.distributions as dist
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.utils.dotbuilder import print_graph
from beanmachine.ppl.world.variable import Variable
from torch import Tensor, tensor


Variables = Dict[RVIdentifier, Variable]


class World(object):
    """
    Represents the world through inference run.

    takes in:
        init_world_likelihood: the likelihood of the initial world being passed
        in.
        init_world: the initial world from which the inference algorithm can
        start from. (helps us to support resumable inference)

    holds:
        variables_: a dict of variables keyed with their function signature.
        log_prob_: joint log probability of the world
        diff_: it is a subset of the variables_ thats includes changes to the
               world variables_.
        diff_log_update_: it is the variable that tracks the update in the diff
                          log probability.

    for instance for model below:

    @sample
    def foo():
        return dist.Bernoulli(torch.tensor(0.1))

    @sample
    def bar():
        if not foo().item():
            return dist.Bernoulli(torch.tensor(0.1))
        else:
            return dist.Bernoulli(torch.tensor(0.9))


    World.variables_ will be:

    defaultdict(<class 'beanmachine.ppl.utils.variable.Variable'>,
    {
     RVIdentifier(function=<function bar at 0x7f6c82b0e488>, arguments=()):
        Variable(
                 distribution=Bernoulli(probs: 0.8999999761581421,
                                        logits: 2.1972243785858154),
                 value=tensor(0.),
                 parent={RVIdentifier(function=<function foo at 0x7f6d343c8bf8>,
                    arguments=())},
                 children=set(),
                 log_prob=tensor(-2.3026)
                ),
     RVIdentifier(function=<function foo at 0x7f6d343c8bf8>, arguments=()):
         Variable(
                  distribution=Bernoulli(probs: 0.10000000149011612,
                                         logits: -2.1972246170043945),
                  value=tensor(0.),
                  parent=set(),
                  children={RVIdentifier(
                                function=<function bar at 0x7f6c82b0e488>,
                                arguments=())},
                  log_prob=tensor(-0.1054)
                 )
     }
    )
    """

    def __init__(
        self,
        # pyre-fixme[9]: init_world_log_prob has type `Tensor`; used as `None`.
        init_world_log_prob: Tensor = None,
        # pyre-fixme[9]: init_world_dict has type `Dict[RVIdentifier, Variable]`;
        #  used as `None`.
        init_world_dict: Dict[RVIdentifier, Variable] = None,
    ):
        # pyre-fixme[6]: Expected `Optional[typing.Callable[[],
        #  Variable[collections._VT]]]` for 1st param but got `Type[Variable]`.
        self.variables_ = defaultdict(Variable)
        self.log_prob_ = tensor(0.0)
        self.observations_ = defaultdict()
        self.reset_diff()
        self.should_transform_ = defaultdict(bool)
        self.should_transform_all_ = False

    def set_all_nodes_transform(self, should_transform: bool) -> None:
        self.should_transform_all_ = should_transform

    def set_transform(self, node, val):
        """
        Enables transform for a given node.

        :param node: the node to enable the transform for
        """
        self.should_transform_[node] = val

    def get_transform(self, node):
        """
        Returns whether transform is enabled for a given node.

        :param node: the node to look up
        :returns: whether the node has transform enabled or not
        """
        return self.should_transform_[node] or self.should_transform_all_

    def __str__(self) -> str:
        return (
            "Variables:\n"
            + "\n".join(
                [str(key) + "=" + str(value) for key, value in self.variables_.items()]
            )
            + "\n\nObservations:\n"
            + "\n".join(
                [
                    str(key) + "=" + str(value.item())
                    for key, value in self.observations_.items()
                ]
            )
            + "\n"
        )

    def to_dot(self) -> str:
        def get_children(rv: RVIdentifier) -> List[Tuple[str, RVIdentifier]]:
            return [("", rv) for rv in self.variables_[rv].children]

        return print_graph(self.variables_.keys(), get_children, str, str)

    def set_observations(self, val: Variables) -> None:
        self.observations_ = val

    def add_node_to_world(self, node: RVIdentifier, var: Variable) -> None:
        """
        Add the node to the world. Since all updates are done through diff_,
        here we will just update diff_.

        :param node: the node signature to be added to world
        :param var: the variable to be added to the world for node
        """
        self.diff_[node] = var

    def update_diff_log_prob(self, node: RVIdentifier) -> None:
        """
        Adds the log update to diff_log_update_

        :param node: updates the diff_log_update_ with the log_prob update of
        the node
        """
        self.diff_log_update_ += self.diff_[node].log_prob - (
            self.variables_[node].log_prob if node in self.variables_ else tensor(0.0)
        )

    def compute_score(self, node_var: Variable) -> Tensor:
        """
        Computes the score of the node plus its children

        :param node_var: the node variable whose score we are going to compute
        :returns: the computed score
        """
        score = node_var.log_prob.clone()
        for child in node_var.children:
            if child is None:
                raise ValueError(f"node {child} is not in the world")
            child_var = self.get_node_in_world_raise_error(child, False)
            score += child_var.log_prob.clone()
        score += node_var.jacobian
        return score

    def get_old_unconstrained_value(self, node: RVIdentifier) -> Optional[Tensor]:
        """
        Looks up the node in the world and returns the old unconstrained value
        of the node.

        :param node: the node to look up
        :returns: old unconstrained value of the node.
        """
        if node in self.variables_:
            if (
                isinstance(self.variables_[node].distribution, dist.Beta)
                and self.variables_[node].proposal_distribution is not None
                and isinstance(
                    self.variables_[node].proposal_distribution.proposal_distribution,
                    dist.Dirichlet,
                )
            ):
                return self.variables_[node].extended_val
            return self.variables_[node].unconstrained_value
        return None

    def get_node_in_world_raise_error(
        self, node: RVIdentifier, to_be_copied: bool = True
    ) -> Variable:
        """
        Get the node in the world, by first looking up diff_, if not available,
        then it can be looked up in variables_, while copying into diff_ and
        returns the new diff_ node (if to_be_copied is true, if not returns
        node's variable in variables_) and if not available in any, returns None

        :param node: node to be looked up in world
        :param to_be_copied: a flag to determine whether add the new node to
        diff_ and start tracking its changes or not.
        :returns: the corresponding node from the world and raises an error if
        node is not available
        """
        node_var = self.get_node_in_world(node, to_be_copied)
        if node_var is None:
            raise ValueError(f"Node {node} is not available in world")

        return node_var

    def get_node_in_world(
        self, node: RVIdentifier, to_be_copied: bool = True
    ) -> Optional[Variable]:
        """
        Get the node in the world, by first looking up diff_, if not available,
        then it can be looked up in variables_, while copying into diff_ and
        returns the new diff_ node (if to_be_copied is true, if not returns
        node's variable in variables_) and if not available in any, returns None

        :param node: node to be looked up in world
        :param to_be_copied: a flag to determine whether add the new node to
        diff_ and start tracking its changes or not.
        :returns: the corresponding node from the world
        """
        if node in self.diff_:
            return self.diff_[node]
        elif node in self.variables_:
            if to_be_copied:
                self.diff_[node] = self.variables_[node].copy()
                return self.diff_[node]
            else:
                return self.variables_[node]

        return None

    def get_number_of_variables(self) -> int:
        return len(self.variables_) - len(self.observations_)

    def contains_in_world(self, node: RVIdentifier) -> bool:
        """
        Looks up both variables_ and diff_ and returns true if node is available
        in any of them, otherwise, returns false

        :param node: node to be looked up in the world
        :returns: true if found else false
        """
        if node in self.diff_ or node in self.variables_:
            return True
        return False

    def get_all_world_vars(self) -> Variables:
        """
        :returns: all variables in the world
        """
        return self.variables_

    def accept_diff(self) -> None:
        """
        If changes in a diff is accepted, world's variables_ are updated with
        their corrseponding diff_ value.
        """
        for node in self.diff_:
            self.variables_[node] = self.diff_[node]

        for node in self.is_delete_:
            if self.is_delete_[node]:
                del self.variables_[node]
        self.log_prob_ += self.diff_log_update_
        self.reset_diff()

    def reset_diff(self) -> None:
        """
        Resets the diff
        """
        # pyre-fixme[6]: Expected `Optional[typing.Callable[[],
        #  Variable[collections._VT]]]` for 1st param but got `Type[Variable]`.
        self.diff_ = defaultdict(Variable)
        self.diff_log_update_ = tensor(0.0)
        self.is_delete_ = defaultdict(bool)

    def start_diff_with_proposed_val(
        self, node: RVIdentifier, proposed_value: Tensor
    ) -> Tensor:
        """
        Starts a diff with new value for node.

        :param node: the node who has a new proposed value
        :param proposed_value: the proposed value for node
        :returns: difference of old and new log probability of the node after
        updating the node value to the proposed value
        """
        # pyre-fixme[6]: Expected `Optional[typing.Callable[[],
        #  Variable[collections._VT]]]` for 1st param but got `Type[Variable]`.
        self.diff_ = defaultdict(Variable)
        var = self.variables_[node].copy()
        old_log_prob = var.log_prob
        var.update_fields(proposed_value, None, self.should_transform_[node])
        var.proposal_distribution = None
        self.diff_[node] = var
        node_log_update = var.log_prob - old_log_prob
        self.diff_log_update_ += node_log_update
        return node_log_update

    def reject_diff(self) -> None:
        """
        Resets the diff_ to an empty dictionary once the diff_ is rejected.
        """
        self.reset_diff()

    def update_children_parents(self, node: RVIdentifier):
        """
        Update the parents that the child no longer depends on.

        :param node: the node whose parents are being evaluated.
        """
        for child in self.diff_[node].children.copy():
            if child not in self.variables_:
                continue

            old_parents = (
                self.variables_[child].parent if child in self.variables_ else set()
            )
            new_parents = self.diff_[child].parent

            dropped_parents = old_parents - new_parents

            for parent in dropped_parents:
                parent_var = self.get_node_in_world(parent)
                # pyre-fixme[16]: `Optional` has no attribute `children`.
                parent_var.children.remove(child)
                if len(parent_var.children) != 0 or parent in self.observations_:
                    continue

                self.is_delete_[parent] = True
                self.diff_log_update_ -= self.variables_[parent].log_prob

                # pyre-fixme[16]: `Optional` has no attribute `parent`.
                ancestors = [(parent, x) for x in parent_var.parent]
                while len(ancestors) > 0:
                    ancestor_child, ancestor = ancestors.pop(0)
                    ancestor_var = self.get_node_in_world(ancestor)
                    ancestor_var.children.remove(ancestor_child)
                    if (
                        len(ancestor_var.children) == 0
                        and ancestor not in self.observations_
                    ):
                        self.is_delete_[ancestor] = True
                        self.diff_log_update_ -= self.variables_[ancestor].log_prob
                        ancestors.extend([(ancestor, x) for x in ancestor_var.parent])

    def create_child_with_new_distributions(
        self, node: RVIdentifier, stack: List[RVIdentifier]
    ) -> Tensor:
        """
        Adds all node's children to diff_ and re-computes their distrbutions
        and log_prob

        :param node: the node whose value was just updated to a proposed value
        and thus its children's distributions are needed to be recomputed.
        :param stack: the inference stack
        :returns: difference of old and new log probability of the immediate
        children of the resampled node.
        """
        old_log_probs = defaultdict()
        new_log_probs = defaultdict()
        for child in self.variables_[node].children:
            child_var = self.get_node_in_world(child)
            if child_var is None:
                continue
            old_log_probs[child] = child_var.log_prob
            child_var.parent = set()
            stack.append(child)
            child_var.distribution = child.function(*child.arguments)
            stack.pop()
            obs_value = (
                self.observations_[child] if child in self.observations_ else None
            )
            child_var.update_fields(
                child_var.value, obs_value, self.should_transform_[child]
            )
            new_log_probs[child] = child_var.log_prob

        self.update_children_parents(node)

        children_log_update = tensor(0.0)
        for node in old_log_probs:
            if node in new_log_probs and not self.is_delete_[node]:
                children_log_update += new_log_probs[node] - old_log_probs[node]
        self.diff_log_update_ += children_log_update
        return children_log_update

    def propose_change(
        self, node: RVIdentifier, proposed_value: Tensor, stack: List[RVIdentifier]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Creates the diff for the proposed change

        :param node: the node who has a new proposed value
        :param proposed_value: the proposed value for node
        :param stack: the inference stack
        :returns: difference of old and new log probability of node's children,
        difference of old and new log probability of world, difference of old
        and new log probability of node
        """
        node_log_update = self.start_diff_with_proposed_val(node, proposed_value)
        children_node_log_update = self.create_child_with_new_distributions(node, stack)
        world_log_update = self.diff_log_update_
        return children_node_log_update, world_log_update, node_log_update
