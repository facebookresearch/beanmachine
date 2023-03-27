# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from beanmachine.facebook.goal_inference.environment import State

from beanmachine.facebook.goal_inference.planner.planner import StateNode


class Gem:
    """A Gem object can be held by the agent

    Arguments/Attributes:
        name: The name of the Gem

    """

    __slots__ = "name"

    def __init__(self, name: str):
        self.name: str = name


@dataclass(frozen=True, eq=True)
class CGemState(State):
    """A State of a continuous gems game is defined by the properties of the environment and the agent

    Arguments/Attributes:

        width: The width of the environment
        height: The height of the environment

        gems: Map from name to Gem for all gems in the problem
        gem_size: The size of the gems. All points with manhattan distance to a gem < gem_size will be considered the gem.

        has: Map from name to Gem for all gems held by agent.
        at: Map from name to location for all gems not held by the agent.

        obstacles: Positions of the center of obstacles
        obstacle_size: The size of the obstacles. All points with manhattan distance to an obstacle < obstacle_size will be blocked.

        agent_size: The size of the agent. All points with manhattan distance to a agent(x,y) < agent_size will be considered the agent.
        x: X-position of Agent
        y: Y-position of Agent
        angle: Current Facing direction of agent. Angle is measured in degrees from rightward direction (0 is same as standard unit circle)

    """

    goal: Tuple[str, str]

    width: float
    height: float

    gems: Dict[str, Gem]
    gem_size: float

    has: Dict[str, Gem]
    at: Dict[str, Tuple[float, float]]

    obstacles: Set[Tuple[float, float]]
    obstacle_size: float

    agent_size: float
    x: float
    y: float
    angle: float

    def __str__(self) -> str:
        """Records the current state of the system as a string
        Returns:
            state_str: A string representation of the state

        """
        state_str = "Current State of Continuous Gems Problem\n”"
        state_str += "Holding\n”"
        state_str += str(list(self.has.keys()))
        state_str += "\n"
        state_str += "On Ground\n"
        state_str += str(list(self.at.keys()))
        state_str += "\n"
        state_str += "X Position"
        state_str += str(round(self.x * 4.0) / 4.0)
        state_str += "\n"
        state_str += "Y Position"
        state_str += str(round(self.y * 4.0) / 4.0)
        state_str += "\n"
        state_str += "Angle"
        state_str += str(round(self.angle, -1))
        return state_str

    def __hash__(self):
        """Computes a hash of the state based on the string representation
        Returns:
            hash: The computed hash of the state
        """
        return hash(str(self))

    def __eq__(self, other: CGemState) -> bool:
        """Modifies the equality comparision by discretizing continuous variables

        Arguments:
            other: The State to compare with

        Returns:
            is_equal: Whether the two states are equal

        """
        if not isinstance(other, CGemState):
            raise (RuntimeError("Tried to compare a CGemState with another object!"))
        x_position = round(self.x * 4.0) / 4.0 == round(other.x * 4.0) / 4.0
        y_position = round(self.y * 4.0) / 4.0 == round(other.y * 4.0) / 4.0
        angle = round(self.angle, -1) == round(other.angle, -1)
        has = self.has.keys() == other.has.keys()
        return x_position and y_position and angle and has


class CGemStateNode(StateNode):

    """
    Defines CGem nodes for graph-based representation of relationship between states.

    Arguments/Attributes:
        executed: Previous operation to get to this point in format [action_name,*args]
        parent_node: Parent to this node
        state: current State

    """

    def __init__(
        self,
        state: CGemState,
        parent_node: Optional[CGemStateNode],
        executed: List[str],
    ):
        self.executed: List[str] = executed
        self.parent_node: Optional[CGemStateNode] = parent_node
        self.state: CGemState = state
