# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
from beanmachine.ppl.experimental.vi.mean_field_variational_approximation import (
    MeanFieldVariationalApproximation,
)
from beanmachine.ppl.legacy.world.diff import Diff
from beanmachine.ppl.legacy.world.diff_stack import DiffStack
from beanmachine.ppl.legacy.world.variable import TransformData, TransformType, Variable
from beanmachine.ppl.legacy.world.world_vars import WorldVars
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils.dotbuilder import print_graph
from beanmachine.ppl.world.base_world import BaseWorld
from torch import Tensor
from torch.distributions import Distribution


Variables = Dict[RVIdentifier, Variable]


class World(BaseWorld):
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

    @bm.random_variable
    def foo():
        return dist.Bernoulli(torch.tensor(0.1))

    @bm.random_variable
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

    vi_dicts: Optional[Callable[[RVIdentifier], MeanFieldVariationalApproximation]]
    params_: Dict[RVIdentifier, nn.Parameter]
    model_to_guide_ids_: Optional[Dict[RVIdentifier, RVIdentifier]]

    def __init__(self):
        self.variables_ = WorldVars()
        self.stack_ = []
        self.observations_ = defaultdict()
        self.reset_diff()
        self.transforms_ = defaultdict(lambda: TransformData(TransformType.NONE, []))
        self.proposer_ = defaultdict(lambda: None)
        self.initialize_from_prior_ = False
        self.maintain_graph_ = True
        self.cache_functionals_ = False
        self.cached_functionals_ = defaultdict()
        self.vi_dicts = None
        self.params_ = {}
        self.model_to_guide_ids_ = None

    def set_initialize_from_prior(self, initialize_from_prior: bool = True):
        """
        Initialize the random variables from their priors.

        :param initialize_from_prior: if True sample from prior else initialize to 0 in
        unconstrained space.
        """
        self.initialize_from_prior_ = initialize_from_prior

    def set_cache_functionals(self, cache_functionals: bool = True):
        """
        Cache any functionals computed in world. maintain_graph_ needs to be False for cache_funtionals
        to be True

        :param cache_functionals: if True cache functionals
        """
        self.cache_functionals_ = cache_functionals

    def get_cache_functionals(self) -> bool:
        """
        returns if the cache_functionals flag is set in the world
        """
        return self.cache_functionals_

    def set_maintain_graph(self, maintain_graph: bool = True):
        """
        Indicate if updates to nodes in graph requires updating it's markov blanket.
        Some inference methds do not require updates to world and node log probs upon proposal.
        chache_functionals needs to be False for maintain_graph to be True
        :param maintain_graph: if True compute log prob updates and markov blankets
        """
        self.maintain_graph_ = maintain_graph

    def set_transforms(
        self,
        func_wrapper,
        transform_type: TransformType,
        transforms: Optional[List] = None,
    ):
        """
        Enables transform for a given node.

        :param node: the node to enable the transform for
        """
        if transform_type == TransformType.CUSTOM and transforms is None:
            raise ValueError("No transform provided for custom transform")
        self.transforms_[func_wrapper] = TransformData(transform_type, transforms)

    def set_all_nodes_transform(
        self, transform_type: TransformType, transforms: Optional[List] = None
    ):
        if transform_type == TransformType.CUSTOM and transforms is None:
            raise ValueError("No transform provided")

        self.transforms_ = defaultdict(
            lambda: TransformData(transform_type, transforms)
        )

    def get_transforms_for_node(self, node):
        """
        Returns whether transform is enabled for a given node.

        :param node: the node to look up
        :returns: whether the node has transform enabled or not
        """
        return self.transforms_[node.wrapper]

    def set_proposer(self, func_wrapper, proposer):
        """
        Sets proposer for a given node

        :param func_wrapper: the function wrapper
        :param proposer: the associated proposer
        """
        self.proposer_[func_wrapper] = proposer

    def set_all_nodes_proposer(self, proposer):
        """
        Sets the default proposers for all nodes

        :param proposer: the default proposer
        """
        self.proposer_ = defaultdict(lambda: proposer)

    def reject_latest_diff(self):
        """
        Reject the top diff.
        """
        if self.diff_stack_.len() == 1:
            self.reject_diff()
        else:
            self.diff_ = self.diff_stack_.remove_last_diff()

    def get_proposer_for_node(self, node):
        """
        Returns the proposer for a given node

        :param node: the node to look up
        :returns: the associate proposer
        """
        return self.proposer_[node.wrapper]

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
                    str(key) + "=" + str(value)
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
        self.diff_stack_.add_node(node, var)

    def get_node_earlier_version(self, node: RVIdentifier) -> Optional[Variable]:
        """
        Get the earlier version of the node in the world.

        :param node: the RVIdentifier of the node to be looked up in the world.
        :returns: the earlier version of the node.
        """
        node_var = self.diff_stack_.get_node_earlier_version(node)
        if node_var is None:
            node_var = self.variables_.get_node(node)
        return node_var

    def update_diff_log_prob(self, node: RVIdentifier) -> None:
        """
        Adds the log update to diff_log_update_

        :param node: updates the diff_log_update_ with the log_prob update of
        the node
        """
        node_var = self.get_node_earlier_version(node)
        new_log_prob = self.diff_stack_.get_node_raise_error(node).log_prob
        if node_var is not None:
            new_log_prob -= node_var.log_prob
        self.diff_stack_.update_log_prob(new_log_prob)

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
            # We just need to read the node in the latest diff.
            child_var = self.get_node_in_world_raise_error(child, False)
            score += child_var.log_prob.clone()
        score += node_var.jacobian
        return score

    def get_old_transformed_value(self, node: RVIdentifier) -> Optional[Tensor]:
        """
        Looks up the node in the world and returns the old transformed value
        of the node.

        :param node: the node to look up
        :returns: old transformed value of the node.
        """
        node_var = self.get_node_earlier_version(node)
        if node_var is not None:
            return node_var.transformed_value
        return None

    def get_old_value(self, node: RVIdentifier) -> Optional[Tensor]:
        """
        Looks up the node in the world and returns the old value of the node.

        :param node: the node to look up
        :returns: old value of the node.
        """
        node_var = self.get_node_earlier_version(node)
        if node_var is not None:
            return node_var.value
        return None

    def get_node_in_world_raise_error(
        self,
        node: RVIdentifier,
        to_be_copied: bool = True,
        to_create_new_diff: bool = False,
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
        node_var = self.get_node_in_world(node, to_be_copied, to_create_new_diff)
        if node_var is None:
            raise ValueError(f"Node {node} is not available in world")

        return node_var

    def get_node_in_world(
        self,
        node: RVIdentifier,
        to_be_copied: bool = True,
        to_create_new_diff: bool = False,
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
        node_var_from_diff = self.diff_stack_.get_node(node)
        if node_var_from_diff:
            if to_create_new_diff:
                node_var_copied = node_var_from_diff.copy()
                self.diff_stack_.add_node(node, node_var_copied)
                return node_var_copied
            return node_var_from_diff
        elif self.variables_.contains_node(node):
            node_var = self.variables_.get_node_raise_error(node)
            if to_be_copied:
                node_var_copied = node_var.copy()
                self.diff_stack_.add_node(node, node_var_copied)
                return node_var_copied
            else:
                return node_var
        return None

    def get_number_of_variables(self) -> int:
        """
        :returns: the number of random variables in the world.
        """
        return self.variables_.len() - len(self.observations_)

    def contains_in_world(self, node: RVIdentifier) -> bool:
        """
        Looks up both variables_ and diff_ and returns true if node is available
        in any of them, otherwise, returns false

        :param node: node to be looked up in the world
        :returns: true if found else false
        """
        return self.diff_stack_.contains_node(node) or self.variables_.get_node(node)

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
        self.diff_stack_.push_changes(self.variables_)
        self.variables_.update_log_prob(self.diff_stack_.diffs_log_prob())
        self.reset_diff()

    def reset_diff(self) -> None:
        """
        Resets the diff and diff stack.
        """
        self.diff_stack_ = DiffStack()
        self.diff_ = self.diff_stack_.diff_stack_[-1]

    def copy(self):
        """
        Returns a copy of the world.
        """
        world_copy = World()
        world_copy.transforms_ = copy.deepcopy(self.transforms_)
        world_copy.proposer_ = copy.deepcopy(self.proposer_)
        return world_copy

    def get_markov_blanket(self, node: RVIdentifier) -> Set[RVIdentifier]:
        """
        Extracts the markov block of a node in the world and we exclude the
        observed random variables.

        :param node: the node for which we'd like to extract the markov blanket
        :returns: the markov blanket of a specific node passed in
        """
        markov_blanket = set()
        node_var = self.get_node_in_world_raise_error(node, False)
        for child in node_var.children:
            if child is None:
                raise ValueError("child is None")
            markov_blanket.add(child)
            child_var = self.get_node_in_world_raise_error(child, False)
            for parent in child_var.parent:
                if parent != node:
                    markov_blanket.add(parent)
        return markov_blanket

    def is_marked_node_for_delete(self, node: RVIdentifier) -> bool:
        """
        Returns whether a node is marked for delete.

        :param node: the node to be looked up in the world.
        :returns: whether the node is marked for delete.
        """
        return self.diff_stack_.is_marked_for_delete(node)

    def get_all_nodes_from_func(self, node_func: str) -> Set[RVIdentifier]:
        """
        Fetches all nodes that have a given node function.

        :param node_func: the node function
        :returns: list of nodes with a given node function
        """
        return self.variables_.get_nodes_by_func(node_func)

    def start_diff_with_proposed_val(
        self, node: RVIdentifier, proposed_value: Tensor, start_new_diff: bool = False
    ) -> Tensor:
        """
        Starts a diff with new value for node.

        :param node: the node who has a new proposed value
        :param proposed_value: the proposed value for node
        :returns: difference of old and new log probability of the node after
        updating the node value to the proposed value
        """
        if not start_new_diff:
            self.reset_diff()
            var = self.variables_.get_node_raise_error(node).copy()
        else:
            self.diff_stack_.add_diff(Diff())
            self.diff_ = self.diff_stack_.top()
            var = self.get_node_in_world_raise_error(node).copy()

        old_log_prob = var.log_prob
        var.update_fields(
            proposed_value.to(old_log_prob.device),
            None,
            self.get_transforms_for_node(node),
            self.get_proposer_for_node(node),
        )
        var.proposal_distribution = None
        self.diff_stack_.add_node(node, var)
        node_log_update = var.log_prob - old_log_prob
        self.diff_stack_.update_log_prob(node_log_update)
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
        for child in self.diff_stack_.get_node_raise_error(node).children.copy():
            if not self.variables_.contains_node(child):
                continue

            new_child_var = self.diff_stack_.get_node(child)
            old_child_var = self.get_node_earlier_version(child)
            if not new_child_var or not old_child_var:
                continue
            new_parents = new_child_var.parent
            old_parents = old_child_var.parent if old_child_var is not None else set()

            dropped_parents = old_parents - new_parents
            for parent in dropped_parents:
                # parent node variable may need to be added to the latest
                # diff, so we may need to call get_node_in_world with
                # to_create_new_diff = True.
                parent_var = self.get_node_in_world(
                    parent, to_create_new_diff=not self.diff_.contains_node(parent)
                )
                # pyre-fixme
                if not parent_var and child not in parent_var.children:
                    continue
                parent_var.children.remove(child)
                if len(parent_var.children) != 0 or parent in self.observations_:
                    continue

                self.diff_stack_.mark_for_delete(parent)
                self.diff_stack_.update_log_prob(
                    -1 * self.variables_.get_node_raise_error(parent).log_prob
                )

                # pyre-fixme[16]: `Optional` has no attribute `parent`.
                ancestors = [(parent, x) for x in parent_var.parent]
                while len(ancestors) > 0:
                    ancestor_child, ancestor = ancestors.pop(0)
                    # ancestor node variable may need to be added to the latest
                    # diff, so we may need to call get_node_in_world with
                    # to_create_new_diff = True.
                    ancestor_var = self.get_node_in_world(
                        ancestor,
                        to_create_new_diff=not self.diff_.contains_node(ancestor),
                    )
                    if ancestor_var is None:
                        continue
                    ancestor_var.children.remove(ancestor_child)
                    if (
                        len(ancestor_var.children) == 0
                        and ancestor not in self.observations_
                    ):
                        self.diff_stack_.mark_for_delete(ancestor)
                        self.diff_stack_.update_log_prob(
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
        # Upto this point, a new diff with node is started, so this will return
        # the node in the latest diff.
        node_var = self.get_node_in_world_raise_error(node)
        for child in node_var.children.copy():
            if child is None:
                continue
            # Children are not yet available in the latest diff, here, we will
            # add children to the latest diff and add them to the stack.
            child_var = self.get_node_in_world_raise_error(
                child, to_create_new_diff=True
            )
            old_log_probs[child] = child_var.log_prob
            child_var.parent = set()
            self.stack_.append(child)
            with self:
                # in this call child is going to be copied over to the latest diff.
                child_var.distribution = child.function(*child.arguments)
            self.stack_.pop()
            obs_value = (
                self.observations_[child] if child in self.observations_ else None
            )
            child_var.update_fields(
                child_var.value,
                obs_value,
                self.get_transforms_for_node(child),
                self.get_proposer_for_node(child),
            )
            new_log_probs[child] = child_var.log_prob

        self.update_children_parents(node)
        graph_update = not self.variables_.contains_node(node) or (
            self.diff_.len()
            > len(self.variables_.get_node_raise_error(node).children) + 1
        )
        children_log_update = torch.zeros((), device=node_var.value.device)
        for node in old_log_probs:
            if node in new_log_probs and not self.diff_.is_marked_for_delete(node):
                children_log_update += new_log_probs[node] - old_log_probs[node]
        self.diff_.update_log_prob(children_log_update)
        return children_log_update, graph_update

    def propose_change_transformed_value(
        self,
        node: RVIdentifier,
        proposed_transformed_value: Tensor,
        allow_graph_update: bool = True,
        start_new_diff: bool = False,
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
        # diff to track the changes is not started yet, so we're just reading
        # the node variable available in world_vars or diff.
        node_var = self.get_node_in_world_raise_error(node, False)
        proposed_value = node_var.inverse_transform_value(proposed_transformed_value)
        return self.propose_change(node, proposed_value, allow_graph_update)

    def update_cached_functionals(self, f, *args) -> Tensor:
        if (f, *args) not in self.cached_functionals_:
            with self:
                self.cached_functionals_[(f, *args)] = f(*args)
        return self.cached_functionals_[(f, *args)]

    def propose_change(
        self,
        node: RVIdentifier,
        proposed_value: Tensor,
        allow_graph_update: bool = True,
        start_new_diff: bool = False,
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
        assert self.maintain_graph_ != self.cache_functionals_
        if not self.maintain_graph_:
            var = self.variables_.get_node_raise_error(node).copy()
            var.update_value(proposed_value)
            # pyre-fixme the inference methods that call this propose_change method but does not require
            # to maintain world would not expect any return values
            return

        node_log_update = self.start_diff_with_proposed_val(
            node, proposed_value, start_new_diff=start_new_diff
        )
        (
            children_node_log_update,
            graph_update,
        ) = self.create_child_with_new_distributions(node)
        if not allow_graph_update and graph_update:
            raise RuntimeError(f"Computation graph changed after proposal for {node}")
        world_log_update = self.diff_.log_prob()
        # diff is already started and changes to world have already happened by
        # the time we get here, so we'd just like to fetch the node from the
        # latest diff.
        diff_node_var = self.get_node_in_world_raise_error(node, False)
        proposed_score = self.compute_score(diff_node_var)
        return (
            children_node_log_update,
            world_log_update,
            node_log_update,
            proposed_score,
        )

    def get_param(self, param: RVIdentifier) -> nn.Parameter:
        "Gets a parameter or initializes it if not found."
        if param not in self.params_:
            self.params_[param] = nn.Parameter(param.function(*param.arguments))
        return self.params_[param]

    def set_params(self, params: Dict[RVIdentifier, nn.Parameter]):
        "Sets the parameters in this World to specified values."
        self.params_ = params

    def update_graph(self, node: RVIdentifier) -> Tensor:
        """
        Updates the parents and children of the node based on the stack

        :param node: the node which was called from StatisticalModel.random_variable()
        """
        assert self.maintain_graph_ != self.cache_functionals_
        if len(self.stack_) > 0:
            # We are making updates to the parent, so we need to call the
            # get_node_in_world_raise_error, we don't need to add the variable
            # to the latest diff because it's been already added there given
            # that it's in the stack.
            self.get_node_in_world_raise_error(self.stack_[-1]).parent.add(node)

        # We are adding the diff manually to the latest diff manually in line
        # 509 and 527.
        node_var = self.get_node_in_world(node, False)
        if node_var is not None:
            if (
                self.maintain_graph_
                and len(self.stack_) > 0
                and self.stack_[-1] not in node_var.children
            ):
                var_copy = node_var.copy()
                var_copy.children.add(self.stack_[-1])
                self.add_node_to_world(node, var_copy)
            return node_var.value

        node_var = Variable(
            # pyre-fixme
            distribution=None,
            # pyre-fixme[6]: Expected `Tensor` for 2nd param but got `None`.
            value=None,
            # pyre-fixme[6]: Expected `Tensor` for 3rd param but got `None`.
            log_prob=None,
            children=set() if len(self.stack_) == 0 else set({self.stack_[-1]}),
            # pyre-fixme[6]: Expected `Transform` for 5th param but got `None`.
            transform=None,
            # pyre-fixme[6]: Expected `Tensor` for 6th param but got `None`.
            transformed_value=None,
            # pyre-fixme[6]: Expected `Tensor` for 7th param but got `None`.
            jacobian=None,
        )

        self.add_node_to_world(node, node_var)
        self.stack_.append(node)
        with self:
            d = node.function(*node.arguments)
            if not isinstance(d, Distribution):
                raise TypeError(
                    "A random_variable is required to return a distribution."
                )
            node_var.distribution = d
        self.stack_.pop()

        obs_value = self.observations_.get(node)

        # resample latents from q
        value = None
        vi_dicts = self.vi_dicts
        model_to_guide_ids = self.model_to_guide_ids_
        if obs_value is None:
            # TODO: messy, consider strategy pattern
            if vi_dicts is not None:
                # mean-field VI
                variational_approx = vi_dicts(node)
                value = variational_approx.rsample((1,)).squeeze()
            elif (
                isinstance(model_to_guide_ids, dict)
                and node not in model_to_guide_ids.values()  # is not a model RV
            ):
                # guide-based VI on non-guide nodes only
                assert (
                    node in model_to_guide_ids
                ), f"Could not find a guide for {node}. VariationalInference requires every latent variable in the model to have a corresponding guide."
                guide_node = model_to_guide_ids[node]
                guide_var = self.get_node_in_world(guide_node)
                if not guide_var:
                    # initialize guide node if missing
                    self.call(guide_node)
                guide_var = self.get_node_in_world_raise_error(guide_node)
                try:
                    value = guide_var.distribution.rsample(torch.Size((1,)))
                except NotImplementedError:
                    value = guide_var.distribution.sample(torch.Size((1,)))

        node_var.update_fields(
            value,
            obs_value,
            self.get_transforms_for_node(node),
            self.get_proposer_for_node(node),
            self.initialize_from_prior_,
        )

        if self.maintain_graph_:
            self.update_diff_log_prob(node)

        return node_var.value

    def update_support(self, node) -> List:
        """
        In place update all the supports of the world variables.
        Only for discrete variables.

        Returns the support of the variable.
        """
        with self:
            pdist = node.function(*node.arguments)
            if not isinstance(pdist, dist.Categorical) and not isinstance(
                pdist, dist.Bernoulli
            ):
                raise ValueError(
                    "Node must be Categorical or Bernoulli, but was %s"
                    % str(node.function)
                )
            var = self.variables_.get_node_raise_error(node)
            var.cardinality = max(len(pdist.probs), var.cardinality)
        return list(range(0, var.cardinality))
