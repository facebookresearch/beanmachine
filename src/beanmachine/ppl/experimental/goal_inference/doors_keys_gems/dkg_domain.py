# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses

from typing import List

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import DKGState
from beanmachine.facebook.goal_inference.environment import Action, Domain


class DKGDomain(Domain):
    """A hard-coded Domain of a doors, keys, and gems game defining predicates (statements that evaluate to True or False) and possible Actions

    Atrributes:

        name: Name specifying the domain is a DKG domain

        Predicates -> True or False (Boolean) statement based on the current state

            wall(x: int,y: int) -> Is there a wall at position x,y
            door(x: int,y: int) -> Is there a door at position x,y
            obstacle(x: int, y:int) -> Is there a door or wall at position x,y
            has(name: str) -> Is the agent holding the object name
            at(name: str, x:int, y:int) -> Is the object name on the ground at position x,y

        Actions (Precondition) -> Modify the existing state given that a precondition is True

            down (If not blocked by boundary or obstacle) -> move state down
            up (If not blocked by boundary or obstacle) -> move state up
            left (If not blocked by boundary or obstacle) -> move state left
            right (If not blocked by boundary or obstacle) -> move state right
            pickup (If Agent is at x,y and item is at x,y) -> Pickup Item
            unlock (If Agent has a key and there is a door in specified direction) -> Unlock Door (losing key)
    """

    def __init__(self):
        self.name = "DKG"

        self.predicates = {}

        self.predicates["wall"] = self._wall
        self.predicates["door"] = self._door
        self.predicates["obstacle"] = self._obstacle
        self.predicates["has"] = self._has
        self.predicates["at"] = self._at

        self.actions = {}

        self.actions["down"] = Action("down", self._down_precondition, self._down)
        self.actions["up"] = Action("up", self._up_precondition, self._up)
        self.actions["right"] = Action("right", self._right_precondition, self._right)
        self.actions["left"] = Action("left", self._left_precondition, self._left)
        self.actions["pickup"] = Action(
            "pickup", self._pickup_precondition, self._pickup
        )
        self.actions["unlock"] = Action(
            "unlock", self._unlock_precondition, self._unlock
        )

    def _wall(self, state: DKGState, x: int, y: int) -> bool:
        """Determines if there is a wall at position (x,y)

        Arguments:
            state: Current state
            x: X-position to test
            y: Y-position to test

        Returns:
            _wall: Whether there is a wall at (x,y)
        """
        return (x, y) in state.walls

    def _door(self, state: DKGState, x: int, y: int) -> bool:
        """Determines if there is a door at position (x,y)

        Arguments:
            state: Current state
            x: X-position to test
            y: Y-position to test

        Returns:
            _door: Whether there is a door at (x,y)
        """
        return (x, y) in state.doors

    def _obstacle(self, state: DKGState, x: int, y: int) -> bool:
        """Determines if there is a door or wall at position (x,y)

        Arguments:
            state: Current state
            x: X-position to test
            y: Y-position to test

        Returns:
            _obstalce: Whether there is a door or wall at (x,y)
        """
        return self._wall(state, x, y) or self._door(state, x, y)

    def _has(self, state: DKGState, name: str) -> bool:
        """Determines if the agent is holding the object name

        Arguments:
            state: State to analyze
            name: Name of the object to look for

        Returns:
            _has: Whether the agent is holding the object name
        """
        return name in state.has

    def _at(self, state: DKGState, name: str, x: int, y: int) -> bool:
        """Determins if there is the object name is a location (x,y)

        Arguments:
            state: State to analyze
            name: Name of the object to look for
            x: X-Position to analyze
            y: Y-position to analyze

        Returns:
            _at: Whether there is the object name is a location (x,y)
        """
        return name in state.at and state.at[name] == (x, y)

    def _down_precondition(self, state: DKGState) -> bool:
        """Determines if agent can move down without colliding with a wall, door, or world boundary

        Arguments:
            state: Current state

        Returns:
            can_mode_down: True if agent can move down
        """

        return state.y > 1 and not self._obstacle(state, state.x, state.y - 1)

    def _down(self, state: DKGState) -> DKGState:
        """Moves an agent downward one

        Arguments:
            state: Current State

        Returns:
            new_state: A new state after moving downward
        """

        return dataclasses.replace(state, y=state.y - 1)

    def _up_precondition(self, state: DKGState) -> bool:
        """Determines if agent can move up without colliding with a wall, door, or world boundary

        Arguments:
            state: Current state

        Returns:
            can_mode_up: True if agent can move up
        """
        return state.y < state.height and not self._obstacle(
            state, state.x, state.y + 1
        )

    def _up(self, state: DKGState) -> DKGState:
        """Moves an agent upward one

        Arguments:
            state: Current State

        Returns:
            new_state: A new state after moving upward
        """
        return dataclasses.replace(state, y=state.y + 1)

    def _left_precondition(self, state: DKGState) -> bool:
        """Determines if agent can move left without colliding with a wall, door, or world boundary

        Arguments:
            state: Current state

        Returns:
            can_mode_left: True if agent can move left
        """
        return state.x > 1 and not self._obstacle(state, state.x - 1, state.y)

    def _left(self, state: DKGState) -> DKGState:
        """Moves an agent left one

        Arguments:
            state: Current State

        Returns:
            new_state: A new state after moving left
        """
        return dataclasses.replace(state, x=state.x - 1)

    def _right_precondition(self, state: DKGState) -> bool:
        """Determines if agent can move right without colliding with a wall, door, or world boundary

        Arguments:
            state: Current state

        Returns:
            can_mode_right: True if agent can move right
        """
        return state.x < state.width and not self._obstacle(state, state.x + 1, state.y)

    def _right(self, state: DKGState) -> DKGState:
        """Moves an agent right one

        Arguments:
            state: Current State

        Returns:
            new_state: A new state after moving right
        """
        return dataclasses.replace(state, x=state.x + 1)

    def _pickup_precondition(self, state: DKGState, item_name: str) -> bool:
        """Determines whether picking up item_name is feasible in the current state
           The item can be picked up if it is on the ground at the same location as the agent

        Arguments:
            state: Current State
            item_name: Item to try and pickup

        Returns:
            _pickup_precondition: Whether the item can be picked up

        """
        return self._at(state, item_name, state.x, state.y) and (
            item_name in state.items
        )

    def _pickup(self, state: DKGState, item_name: str) -> DKGState:
        """Picks up an item with name item_name

        Arguments:
            state: Current State
            item_name: Item to pickup

        Returns:
            new_state: A new state holding the specified item

        """
        new_has = state.has.copy()
        new_at = state.at.copy()
        new_has[item_name] = state.items[item_name]
        new_at.pop(item_name)
        return dataclasses.replace(state, has=new_has, at=new_at)

    def _unlock_precondition(
        self, state: DKGState, key_name: str, direction_name: str
    ) -> bool:
        """Determines whether a door in direction direction_name can be unlocked with key key_name
           The agent must be holding the key key_name
           The door must also be at a location specified by the direction

        Arguments:
            state: Current State
            key_name: Key to use to unlock door
            direction_name: Direction of door

        Returns:
            _unlock_precondition: Whether the door can be unlocked

        """
        return (
            key_name in state.has
            and key_name in state.keys
            and self._door(
                state,
                state.x + state.xdiff[direction_name],
                state.y + state.ydiff[direction_name],
            )
        )

    def _unlock(self, state: DKGState, key_name: str, direction_name: str) -> DKGState:
        """Unlocks a door with key key_name
        Removes the door from the game
        Removes the used key from the game

        Arguments:
            state: Current State
            key_name: Key to use to unlock door
            direction_name: Direction of the door to unlock

        Returns:
            new_state: A new state with the door unlocked and the key removed

        """
        new_items = state.items.copy()
        new_keys = state.keys.copy()
        new_has = state.has.copy()
        new_doors = state.doors.copy()
        new_items.pop(key_name)
        new_keys.pop(key_name)
        new_has.pop(key_name)
        new_doors.remove(
            (
                state.x + state.xdiff[direction_name],
                state.y + state.ydiff[direction_name],
            )
        )
        return dataclasses.replace(
            state, items=new_items, keys=new_keys, has=new_has, doors=new_doors
        )

    def get_possible_actions(self, state: DKGState) -> List[List[str]]:
        """Determines list of actions whose preconditions are met and will execute

        Arguments:
            state: Current state
        Returns:
            possible_actions: A list of possible actions for the next time step
        """
        possible_actions = []
        if self.is_action_possible(state, "right"):
            possible_actions.append(["right"])
        if self.is_action_possible(state, "up"):
            possible_actions.append(["up"])
        if self.is_action_possible(state, "left"):
            possible_actions.append(["left"])
        if self.is_action_possible(state, "down"):
            possible_actions.append(["down"])

        for name in state.at:
            if self.is_action_possible(state, "pickup", name):
                possible_actions.append(["pickup", name])

        for name in state.has:
            for direction in state.directions:
                if self.is_action_possible(state, "unlock", name, direction):
                    possible_actions.append(["unlock", name, direction])

        return possible_actions

    def is_action_possible(self, state: DKGState, action_name: str, *args) -> bool:
        """Evaluates whether the predicate of the action is satisfied

        Arguments:
            state: Current State to analyze
            action_name: Action being attempted
            *args: Parameters of the action

        Returns:
            is_action_possible: Whether the predicate of the action is satisfied

        """
        return self.actions[action_name].check_precondition(state, *args)

    def evaluate_goal(self, state: DKGState) -> bool:
        """Evaluates whether the current goal is achieved

        Arguments:
            state: Current State to analyze

        Returns:
            is_goal_achieved: Whether the current goal is achieved
        """
        return self.predicates[state.goal[0]](state, state.goal[1])
