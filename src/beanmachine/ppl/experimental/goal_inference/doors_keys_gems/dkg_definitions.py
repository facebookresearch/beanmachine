# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from beanmachine.facebook.goal_inference.environment import State

from beanmachine.facebook.goal_inference.planner.planner import StateNode


class Item:
    """An Item object can be held by the agent

    Arguments/Attributes:
        name: The name of the Item
    """

    __slots__ = "name"

    def __init__(self, name: str):
        self.name: str = name


class Key(Item):
    """A Key is an Item that can open doors

    Arguments/Attributes:
        name: The name of the Key
    """

    pass


class Gem(Item):
    """A Gem is an Item that can be the goal

    Arguments/Attributes:
        name: The name of the Gem
    """

    pass


class Direction:
    """A Direction determines the possible interactions with the environment [e.g. movement, unlocking doors]

    Arguments/Attributes:
        name: The name of the Direction
    """

    __slots__ = "name"

    def __init__(self, name: str):
        self.name: str = name


@dataclass(frozen=True, eq=True)
class DKGState(State):
    """A State of a doors, keys, and gems game is defined by the properties of the environment and the Agent

    Arguments/Attributes:

        width: The width of the environment
        height: The height of the environment
        xdiff: Movement/Interaction along x-axis when given a direction
        ydiff: Movement/Interaction along y-axis when given a direction

        doors: Positions of doors
        walls: Positions of walls

        items: Map from name to Item for all Items remaining in game
        directions: Map from name to Direction
        keys: Map from name to Key for all Keys remaining in game
        gems: Map from name to Gem for all Gems remaining in game

        has: Map from name to Item for all Items held by Agent
        at: Map from name to Location for all Items not held by the Agent

        x: X-position of Agent
        y: Y-position of Agent

    """

    goal: Tuple[str, str]

    width: int
    height: int
    xdiff: Dict[str, int]
    ydiff: Dict[str, int]

    doors: Set[(Tuple[int, int])]
    walls: Set[(Tuple[int, int])]

    items: Dict[str, Item]
    directions: Dict[str, Direction]
    keys: Dict[str, Key]
    gems: Dict[str, Gem]

    has: Dict[str, Item]
    at: Dict[str, Tuple[int, int]]

    x: int
    y: int

    def __str__(self) -> str:
        """Records the current state of the system as a string
        Returns:
            state_str: A string representation of the state
        """
        state_str = "Current State of DKG Problem\n”"
        state_str += "Holding\n”"
        state_str += str(list(self.has.keys()))
        state_str += "\n"
        state_str += "On Ground\n"
        state_str += str(list(self.at.keys()))
        state_str += "\n"
        state_str += "X Position"
        state_str += str(self.x)
        state_str += "\n"
        state_str += "Y Position"
        state_str += str(self.y)
        return state_str

    def __hash__(self):
        """Computes a hash of the state based on the string representation
        Returns:
            hash: The computed hash of the state
        """
        return hash(str(self))


class DKGStateNode(StateNode):

    """
    Defines DKG nodes for graph-based representation of relationship between states.

    Arguments/Attributes:

        executed: Previous operation to get to this point in format [action_name,*args]
        parent_node: Parent to this node
        state: current State

    """

    def __init__(
        self,
        state: DKGState,
        parent_node: Optional[DKGStateNode],
        executed: List[str],
    ):
        self.executed: List[str] = executed
        self.parent_node: Optional[DKGStateNode] = parent_node
        self.state: DKGState = state
