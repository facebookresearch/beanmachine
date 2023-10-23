# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from itertools import count

from queue import PriorityQueue

from typing import Callable, Dict, Tuple

from beanmachine.facebook.goal_inference.environment import Domain, State
from beanmachine.facebook.goal_inference.planner.planner import Planner, StateNode


class AstarPlanner(Planner):

    """
    Obtains an A* solution to a problem given a heuristic.

    Arguments:
        domain: Domain that encodes the rules of the world
        heuristic: Heuristic for A* Algorithm

    Attributes:

        domain: Domain that encodes the rules of the world
        visited_nodes: Record of nodes visited during planning
        heuristic: Heuristic for A* Algorithm
        cost: Record of cost to reach visited States
        q: Priority queue for A* algorithm
        unique: StateNode Id to avoid collisions in Priority Queue (StateNodes cannot be compared)

    """

    def __init__(self, domain: Domain, heuristic: Callable):
        super().__init__(domain)
        self.heuristic: Callable = heuristic
        self.cost: Dict[State, int] = {}
        self.q: PriorityQueue[Tuple[int, int, StateNode]] = PriorityQueue()
        self.unique: count[int] = count()

    def is_empty(self) -> bool:
        """Determines whether the priority queue is empty

        Returns:
            is_empty: Whether the astar datastructure is empty
        """
        return self.q.empty()

    def reset(self) -> None:
        """Clears the datatructures associated with this AstarPlanner"""
        self.cost = {}
        self.q = PriorityQueue()
        self.visited_nodes = []

    def get_next_node(self) -> StateNode:
        """Get next StateNode to evaluate

        Returns:
            The next node to evaluate

        """
        curr_prioirity, curr_count, curr_node = self.q.get()
        return curr_node

    def add_node(self, new_node) -> None:
        """Add a node to Astar datastructure

        Arguments:
            new_node: The node to be potentially explored
        """
        if new_node.parent_node is None:
            self.cost[new_node.state] = 0
            self.q.put((0, next(self.unique), new_node))
        else:
            new_cost = self.cost[new_node.parent_node.state] + 1
            if new_node.state not in self.cost or new_cost < self.cost[new_node.state]:
                self.cost[new_node.state] = new_cost
                priority = new_cost + self.heuristic(new_node)
                self.q.put((priority, next(self.unique), new_node))
