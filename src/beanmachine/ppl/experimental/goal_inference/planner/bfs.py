# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from collections import deque

from typing import Deque, Set

from beanmachine.facebook.goal_inference.environment import Domain, State
from beanmachine.facebook.goal_inference.planner.planner import Planner, StateNode


class BFSPlanner(Planner):

    """
    Obtains a Breadth First Search solution to a problem.

    Arguments:
        domain: Domain that encodes the rules of the world

    Attributes:

        domain: Domain that encodes the rules of the world
        visited_nodes: Record of nodes visited during planning
        queue: Maintains queue of StateNodes for BFS
        visited: Maintains a record of visited states

    """

    def __init__(self, domain: Domain):
        super().__init__(domain)
        self.deque: Deque[StateNode] = deque()
        self.visited: Set[State] = set()

    def is_empty(self) -> bool:
        """Evaluates to True if the bfs datastructure is empty

        Returns:
            is_empty: Whether the bfs datastructure is empty

        """
        return not bool(self.deque)

    def reset(self) -> None:
        """Clears the datatructures associated with this BFSPlanner"""
        self.deque = deque()
        self.visited_nodes = []

    def get_next_node(self) -> StateNode:
        """Get next StateNode to evaluate

        Returns:
            The next node to evaluate

        """
        return self.deque.popleft()

    def add_node(self, new_node: StateNode) -> None:
        """Add a node to BFS datastructure

        Arguments:
            new_nodes: The node to be potentially explored

        """
        if new_node.state not in self.visited:
            self.visited.add(new_node.state)
            self.deque.append(new_node)
