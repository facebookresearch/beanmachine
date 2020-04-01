# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.distributions as dist
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.utils.dotbuilder import print_graph
from beanmachine.ppl.world.diff import Diff
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world_vars import WorldVars
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
        self.variables_ = WorldVars()
        self.stack_ = []
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
                [
                    str(key) + "=" + str(value)
                    for key, value in self.variables_.vars().items()
                ]
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
            # pyre-fixme
            return [
                ("", rv) for rv in self.variables_.get_node_raise_error(rv).children
            ]

        # pyre-fixme
        return print_graph(self.variables_.vars().keys(), get_children, str, str)

    def set_observations(self, val: Dict) -> None:
        """
        Sets the observations in the world

        :param val: dict of observations
        """
        self.observations_ = val

    def add_node_to_world(self, node: RVIdentifier, var: Variable) -> None:
        """
        Add the node to the world. Since all updates are done through diff_,
        here we will just update diff_.

        :param node: the node signature to be added to world
        :param var: the variable to be added to the world for node
        """
        self.diff_.add_node(node, var)

    def update_diff_log_prob(self, node: RVIdentifier) -> None:
        """
        Adds the log update to diff_log_update_

        :param node: updates the diff_log_update_ with the log_prob update of
        the node
        """
        node_var = self.variables_.get_node(node)
        self.diff_.update_log_prob(
            self.diff_.get_node_raise_error(node).log_prob
            - (node_var.log_prob if node_var is not None else tensor(0.0))
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
        if self.variables_.contains_node(node):
            node_var = self.variables_.get_node_raise_error(node)
            if (
                isinstance(node_var.distribution, dist.Beta)
                and node_var.proposal_distribution is not None
                and isinstance(
                    node_var.proposal_distribution.proposal_distribution, dist.Dirichlet
                )
            ):
                return node_var.extended_val
            return node_var.unconstrained_value
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
        if self.diff_.contains_node(node):
            return self.diff_.get_node(node)
        elif self.variables_.contains_node(node):
            node_var = self.variables_.get_node_raise_error(node)
            if to_be_copied:
                node_var_copied = node_var.copy()
                self.diff_.add_node(node, node_var_copied)
                return node_var_copied
            else:
                return node_var
        return None

    def get_number_of_variables(self) -> int:
        return self.variables_.len() - len(self.observations_)

    def contains_in_world(self, node: RVIdentifier) -> bool:
        """
        Looks up both variables_ and diff_ and returns true if node is available
        in any of them, otherwise, returns false

        :param node: node to be looked up in the world
        :returns: true if found else false
        """
        return self.diff_.contains_node(node) or self.variables_.get_node(node)

    def get_all_world_vars(self) -> Variables:
        """
        :returns: all variables in the world
        """
        return self.variables_.vars()

    def accept_diff(self) -> None:
        """
        If changes in a diff is accepted, world's variables_ are updated with
        their corrseponding diff_ value.
        """
        for node in self.diff_.vars():
            node_var = self.diff_.get_node_raise_error(node)
            self.variables_.add_node(node, node_var)

        for node, to_be_deleted in self.diff_.to_be_deleted_vars().items():
            if to_be_deleted:
                self.variables_.delete(node)
        self.variables_.update_log_prob(self.diff_.log_prob())
        self.reset_diff()

    def reset_diff(self) -> None:
        """
        Resets the diff
        """
        self.diff_ = Diff()

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
        self.reset_diff()
        var = self.variables_.get_node_raise_error(node).copy()
        old_log_prob = var.log_prob
        var.update_fields(proposed_value, None, self.should_transform_[node])
        var.proposal_distribution = None
        self.diff_.add_node(node, var)
        node_log_update = var.log_prob - old_log_prob
        self.diff_.update_log_prob(node_log_update)
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
        for child in self.diff_.get_node_raise_error(node).children.copy():
            if not self.variables_.contains_node(child):
                continue

            old_child_var = self.variables_.get_node(child)
            old_parents = old_child_var.parent if old_child_var is not None else set()
            new_child_var = self.diff_.get_node(child)
            new_parents = new_child_var.parent

            dropped_parents = old_parents - new_parents
            for parent in dropped_parents:
                parent_var = self.get_node_in_world(parent)
                # pyre-fixme
                if child not in parent_var.children:
                    continue
                parent_var.children.remove(child)
                if len(parent_var.children) != 0 or parent in self.observations_:
                    continue

                self.diff_.mark_for_delete(parent)
                self.diff_.update_log_prob(
                    -1 * self.variables_.get_node_raise_error(parent).log_prob
                )

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
                        self.diff_.mark_for_delete(ancestor)
                        self.diff_.update_log_prob(
                            -1 * self.variables_.get_node_raise_error(ancestor).log_prob
                        )
                        ancestors.extend([(ancestor, x) for x in ancestor_var.parent])

    def create_child_with_new_distributions(
        self, node: RVIdentifier
    ) -> Tuple[Tensor, bool]:
        """
        Adds all node's children to diff_ and re-computes their distrbutions
        and log_prob

        :param node: the node whose value was just updated to a proposed value
        and thus its children's distributions are needed to be recomputed.
        :returns: difference of old and new log probability of the immediate
        children of the resampled node, and flag indicating if parents of children
        have changed
        """
        old_log_probs = defaultdict()
        new_log_probs = defaultdict()
        node_var = self.get_node_in_world_raise_error(node)
        for child in node_var.children.copy():
            if child is None:
                continue
            child_var = self.get_node_in_world_raise_error(child)
            old_log_probs[child] = child_var.log_prob
            child_var.parent = set()
            self.stack_.append(child)
            child_var.distribution = child.function(*child.arguments)
            self.stack_.pop()
            obs_value = (
                self.observations_[child] if child in self.observations_ else None
            )
            child_var.update_fields(
                child_var.value, obs_value, self.should_transform_[child]
            )
            new_log_probs[child] = child_var.log_prob

        self.update_children_parents(node)
        graph_update = (
            self.diff_.len()
            > len(self.variables_.get_node_raise_error(node).children) + 1
        )
        children_log_update = tensor(0.0)
        for node in old_log_probs:
            if node in new_log_probs and not self.diff_.is_marked_for_delete(node):
                children_log_update += new_log_probs[node] - old_log_probs[node]
        self.diff_.update_log_prob(children_log_update)
        return children_log_update, graph_update

    def propose_change_unconstrained_value(
        self,
        node: RVIdentifier,
        proposed_unconstrained_value: Tensor,
        allow_graph_update=True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Creates the diff for the proposed change

        :param node: the node who has a new proposed value
        :param proposed_value: the proposed value for node
        :param allow_graph_update: allow parents of node to update
        :returns: difference of old and new log probability of node's children,
        difference of old and new log probability of world, difference of old
        and new log probability of node
        """
        node_var = self.get_node_in_world_raise_error(node, False)
        proposed_value = node_var.transform_from_unconstrained_to_constrained(
            proposed_unconstrained_value
        )
        return self.propose_change(node, proposed_value, allow_graph_update)

    def propose_change(
        self, node: RVIdentifier, proposed_value: Tensor, allow_graph_update=True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Creates the diff for the proposed change

        :param node: the node who has a new proposed value
        :param proposed_value: the proposed value for node
        :param allow_graph_update: allow parents of node to update
        :returns: difference of old and new log probability of node's children,
        difference of old and new log probability of world, difference of old
        and new log probability of node, new score of world
        """
        node_log_update = self.start_diff_with_proposed_val(node, proposed_value)
        children_node_log_update, graph_update = self.create_child_with_new_distributions(
            node
        )
        if not allow_graph_update and graph_update:
            raise RuntimeError(f"Computation graph changed after proposal for {node}")
        world_log_update = self.diff_.log_prob()
        diff_node_var = self.get_node_in_world_raise_error(node, False)
        proposed_score = self.compute_score(diff_node_var)
        return (
            children_node_log_update,
            world_log_update,
            node_log_update,
            proposed_score,
        )

    def update_graph(self, node: RVIdentifier) -> Tensor:
        """
        Updates the parents and children of the node based on the stack

        :param node: the node which was called from StatisticalModel.sample()
        """
        if len(self.stack_) > 0:
            self.get_node_in_world_raise_error(self.stack_[-1]).parent.add(node)

        node_var = self.get_node_in_world(node, False)
        if node_var is not None:
            if len(self.stack_) > 0 and self.stack_[-1] not in node_var.children:
                var_copy = node_var.copy()
                var_copy.children.add(self.stack_[-1])
                self.add_node_to_world(node, var_copy)
            return node_var.value

        node_var = Variable(
            # pyre-fixme
            distribution=None,
            value=None,
            log_prob=None,
            parent=set(),
            children=set() if len(self.stack_) == 0 else set({self.stack_[-1]}),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=None,
            transforms=None,
            unconstrained_value=None,
            jacobian=None,
        )

        self.add_node_to_world(node, node_var)

        self.stack_.append(node)
        node_var.distribution = node.function(*node.arguments)
        self.stack_.pop()

        obs_value = self.observations_[node] if node in self.observations_ else None
        node_var.update_fields(None, obs_value, self.get_transform(node))
        self.update_diff_log_prob(node)

        return node_var.value
