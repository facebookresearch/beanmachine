# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from beanmachine.facebook.goal_inference.environment import Domain, State


@dataclass
class StateNode:

    """
    Defines nodes for graph-based representation of relationship between states.

    Arguments/Attributes:
        state: Current State
        parent_node: Parent to this node
        executed: Previous operation to get to this point in format [action_name,*args]
    """

    state: State
    parent_node: Optional[StateNode]
    executed: List[str]


@dataclass()
class Plan:
    """
    Defines a sequence of actions in string form (e.g. ["unlock","key1","right"]) and (expected) states

    Arguments/Attributes:
        actions: A sequence of actions to perform
        states: Expected states following actions. e.g. states[i+1] is predicted state after actions[i]
        visited_nodes: Nodes visited during the planning process
        budget: The search budget for creating this plan
    """

    states: List[State]
    actions: List[List[str]]
    visited_nodes: List[StateNode]
    budget: int


class Planner(ABC):
    """A Planner finds sequences of actions to solve a problem

    Arguments:
        domain: The domain that determines the rules of the environment.

    Attributes:
        domain: The domain that determines the rules of the environment.
        visited_nodes: A record of the nodes visited during planning.
    """

    def __init__(self, domain: Domain):
        self.domain: Domain = domain
        self.visited_nodes: List[StateNode] = []

    def generate_plan(
        self, initial_state: State, budget: float = math.inf
    ) -> Tuple[Plan, bool]:
        """Searches for a plan to solve the problem given budgetary constraints
           Determines whether a solution has been reached

        Arguments:
            initial_state: The initial_state of the search
            budget: The number of nodes that can be evaluated when formulating a plan

        Returns:
            plan: The Plan found by the search
            solved: Whether the plan reached the goal

        """
        self.reset()
        init_node = StateNode(initial_state, None, [])
        self.add_node(init_node)
        search_count = 0
        curr_node = init_node
        goal_achieved = self.domain.evaluate_goal(init_node.state)
        while search_count < budget:
            curr_node = self.get_next_node()
            self.visited_nodes.append(curr_node)
            search_count += 1
            goal_achieved = self.domain.evaluate_goal(curr_node.state)
            # Goal is achieved - end planning
            if goal_achieved:
                break
            poss_actions = self.domain.get_possible_actions(curr_node.state)
            for act in poss_actions:
                next_state = self.domain.execute(curr_node.state, *act)
                new_node = StateNode(next_state, executed=act, parent_node=curr_node)
                self.add_node(new_node)
            # Hit a dead end - end planning
            if self.is_empty():
                break

        return (
            # pyre-fixme[19]: Expected 4 positional arguments.
            Plan(*get_execution_path(curr_node), self.visited_nodes, search_count),
            goal_achieved,
        )

    @abstractmethod
    def reset(self) -> None:
        """Clears the datatructures associated with this Planner"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Evaluates to True if the planning datastructure is empty

        Returns:
            is_empty: Whether the planning datastructure is empty
        """
        pass

    @abstractmethod
    def get_next_node(self) -> StateNode:
        """Get next StateNode to evaluate

        Returns:
            The next node to evaluate

        """
        pass

    @abstractmethod
    def add_node(self, new_node: StateNode) -> None:
        """Add a node to potentially be explored

        Arguments:
            new_node: The node to be potentially explored
        """
        pass


def get_execution_path(
    curr_node: Optional[StateNode],
) -> Tuple[List[State], List[List[str]]]:
    """
    Determines All States/Actions to reach the current node.
    Each action is in format [action_name,*args]
    acts[i] is the transition between state[i] and state[i+1]

    Arguments:
        curr_node: The current node whose path is being analyzed

    Returns:
        states: States on path to current node
        actions: Actions on path to current node
    """
    acts = []
    states = []
    while curr_node.parent_node is not None:
        acts.append(curr_node.executed)
        states.append(curr_node.state)
        curr_node = curr_node.parent_node
    states.append(curr_node.state)
    acts.reverse()
    states.reverse()
    return states, acts


def get_plan_from_actions(
    domain: Domain, state: State, actions: List[List[str]]
) -> Plan:
    """Determines the plan that results from a series of actions

    Arguments:
        domain: The domain of the world
        state: Initial State
        actions: Series of actions that will be performed

    Returns:
        Plan: The plan that results from the actions

    """
    states = []
    states.append(state)
    curr_state = state
    for act in actions:
        next_state = domain.execute(curr_state, *act)
        states.append(next_state)
        curr_state = next_state
    return Plan(states, actions, [], 0)
