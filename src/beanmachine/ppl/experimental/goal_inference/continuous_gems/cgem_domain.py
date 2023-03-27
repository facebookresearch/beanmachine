# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses
import math

from typing import Any, List

import numpy as np

from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemState,
)
from beanmachine.facebook.goal_inference.continuous_gems.cgem_utils import (
    check_intersection,
)
from beanmachine.facebook.goal_inference.environment import Action, Domain

from beanmachine.facebook.goal_inference.utils import manhattan_distance


class CGemDomain(Domain):
    """A hard-coded Domain of a continuous gems problem defining predicates (statements that evaluate to True or False) and possible actions

    Atrributes:

        name: Name specifying the domain is a Continuous Gems domain

        Predicates -> True or False (Boolean) statements based on the current state

            obstacle(x: int, y:int) -> Is there an obstacle at position x,y
            has(name: str) -> Is the agent holding the object referred to by name
            at(name: str, x:float, y:float) -> Is the object referred to by name on the ground at position x,y

        Actions (Precondition) -> Modify the existing state given that a precondition is True

            rotate(angle) -> rotate the direction the agent is facing angle degrees
            move(distance) -> move forward distance in the direction the agent is facing
            pickup(gem_name) -> pickup gem_name

    """

    def __init__(self):
        self.name = "Continuous Gems"

        self.predicates = {}

        self.predicates["obstacle"] = self._obstacle
        self.predicates["has"] = self._has
        self.predicates["at"] = self._at

        self.actions = {}

        self.actions["rotate"] = Action(
            "rotate", self._rotate_precondition, self._rotate
        )
        self.actions["move"] = Action("move", self._move_precondition, self._move)
        self.actions["pickup"] = Action(
            "pickup", self._pickup_precondition, self._pickup
        )

    def _obstacle(self, state: CGemState, x: float, y: float) -> bool:
        """Determines if there is an obstacle at position (x,y)

        Arguments:
            state: State to analyze
            x: X-position to analyze
            y: Y-position to analyze

        Returns:
            _obstacle: Whether there is an obstacle at position (x,y)
        """
        for obstacle_position in state.obstacles:
            if manhattan_distance(obstacle_position, (x, y)) < state.obstacle_size:
                return True
        return False

    def _has(self, state: CGemState, name: str) -> bool:
        """Determines if the agent is holding the object name

        Arguments:
            state: State to analyze
            name: Name of the object to look for

        Returns:
            _has: Whether the agent is holding the object name
        """
        return name in state.has

    def _at(self, state: CGemState, name: str, x: float, y: float) -> bool:
        """Determins if there is the object name is a location (x,y)

        Arguments:
            state: State to analyze
            name: Name of the object to look for
            x: X-Position to analyze
            y: Y-position to analyze

        Returns:
            _at: Whether there is the object name is a location (x,y)

        """

        return (
            name in state.at
            and manhattan_distance(state.at[name], (x, y)) < state.gem_size
        )

    def _rotate_precondition(self, state: CGemState, rotation_angle: float) -> bool:
        """Determines whether a rotation is feasible in the current state

        Arguments:
            state: The current state
            rotation_angle: The size of the rotation in degrees

        Returns:
            true: Rotations are always allowed

        """
        return True

    def _rotate(self, state: CGemState, rotation_angle: float) -> CGemState:
        """Rotates an agent by rotation_angle degrees

        Arguments:
            state: Current State
            rotation_angle: How far to rotate the agent's orientation

        Returns:
            new_state: A new state with a modified angle

        """
        return dataclasses.replace(state, angle=(state.angle + rotation_angle) % 360.0)

    def _move_precondition(self, state: CGemState, distance: float) -> bool:
        """Determines whether a movement is feasible in the current state
           The move is discretized into 10 substeps. Each substep is checked to ensure it does not collide with obstacles

        Arguments:
            state: Current State
            distance: The size of the step forward

        Returns:
            can_move: True if the movement does not collide with obstacles

        """
        curr_x = state.x
        curr_y = state.y
        # Check for collisions with obstacles along the possible path
        for _step in range(10):
            # Move forward based on agent's direction
            curr_x += 0.1 * distance * math.cos(math.radians(state.angle))
            curr_y += 0.1 * distance * math.sin(math.radians(state.angle))
            # Check for collisions with obstacles
            for obstacle in state.obstacles:
                if check_intersection(
                    obstacle, state.obstacle_size, (curr_x, curr_y), state.agent_size
                ):
                    return False
            if (
                curr_x - state.agent_size < 0.0
                or curr_x + state.agent_size > state.width
                or curr_y - state.agent_size < 0.0
                or curr_y + state.agent_size > state.height
            ):
                return False
        return True

    def _move(self, state: CGemState, distance: float) -> CGemState:
        """Moves an agent forward by distance

        Arguments:
            state: Current State
            distance: How far to move the agent forward

        Returns:
            new_state: A new state with a modified location

        """
        new_x = state.x + distance * math.cos(math.radians(state.angle))
        new_y = state.y + distance * math.sin(math.radians(state.angle))
        return dataclasses.replace(state, x=new_x, y=new_y)

    def _pickup_precondition(self, state: CGemState, gem_name: str):
        """Determines whether picking up gem_name is feasible in the current state
           A gem can be pickup if the agent's rectangle overlaps with the gem's rectangle

        Arguments:
            state: Current State
            gem_name: Gem to try and pickup

        Returns:
            _pickup_precondition: Whether the gem can be picked up

        """
        return gem_name in state.at and check_intersection(
            state.at[gem_name], state.gem_size, (state.x, state.y), state.agent_size
        )

    def _pickup(self, state: CGemState, gem_name: str) -> CGemState:
        """Picks up a gem with name gem_name

        Arguments:
            state: Current State
            gem_name: Gem to pickup

        Returns:
            new_state: A new state holding the specified gem

        """
        new_has = state.has.copy()
        new_at = state.at.copy()
        new_has[gem_name] = state.gems[gem_name]
        new_at.pop(gem_name)
        return dataclasses.replace(state, has=new_has, at=new_at)

    def get_possible_actions(self, state: CGemState) -> List[List[Any]]:
        """Determines list of actions whose preconditions are met and will execute

        Next possible actions include:
        (1) A rotation sample from a Uniform distribution [0.0,360.0)
        (2) A forward move with distance drawn from Uniform(0,1) (If possible)
        (3) Any possible pickups

        Arguments:
            state: Current state
        Returns:
            possible_actions: A list of possible actions for the next time step
        """
        possible_actions = []
        rotation = 360 * np.random.rand()
        if self.is_action_possible(state, "rotate", rotation):
            possible_actions.append(["rotate", rotation])
        step = np.random.rand()
        if self.is_action_possible(state, "move", step):
            possible_actions.append(["move", step])

        for name in state.at:
            if self.is_action_possible(state, "pickup", name):
                possible_actions.append(["pickup", name])

        return possible_actions

    def is_action_possible(self, state: CGemState, action_name: str, *args) -> bool:
        """Evaluates whether the predicate of the action is satisfied

        Arguments:
            state: Current State to analyze
            action_name: Action being attempted
            *args: Parameters of the action

        Returns:
            is_action_possible: Whether the predicate of the action is satisfied

        """
        return self.actions[action_name].check_precondition(state, *args)

    def evaluate_goal(self, state: CGemState) -> bool:
        """Evaluates whether the current goal is achieved

        Arguments:
            state: Current State to analyze

        Returns:
            is_goal_achieved: Whether the current goal is achieved

        """
        return self.predicates[state.goal[0]](state, state.goal[1])
