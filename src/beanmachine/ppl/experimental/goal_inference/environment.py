# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Tuple


class Action:
    """An Action has a precondition that must be true before being executed on a state.

    Public Atrributes:

    name: The name of the action
    precondition: Evaluates whether a state meets criteria for effect to execute
    effect: Defines change in state when the action is performed

    """

    def __init__(self, name: str, precondition: Callable, effect: Callable):
        self.name: str = name
        self.precondition: Callable = precondition
        self.effect: Callable = effect

    def check_precondition(self, state: State, *args) -> bool:
        """Checks if the precondition evaluates to true"""
        return self.precondition(state, *args)

    def execute(self, state: State, *args) -> State:
        """execute: Changes the state according to the effect if the precondition is met"""
        if self.check_precondition(state, *args):
            return self.effect(state, *args)
        else:
            return state


@dataclass(frozen=True, eq=True)
class State(ABC):
    goal: Tuple[str, ...]


class Domain(ABC):

    __slots__ = "actions", "predicates"

    def __init__(self):
        self.actions: Dict[str, Action]
        self.predicates: Dict[str, Callable]

    def execute(self, state: State, action_name: str, *args):
        return self.actions[action_name].execute(state, *args)

    @abstractmethod
    def get_possible_actions(self, state) -> List[List[Any]]:
        """Gets All Possible Next Actions"""
        pass

    @abstractmethod
    def is_action_possible(self, state, action_name, *args) -> bool:
        """Determines if an action is possible"""
        pass

    @abstractmethod
    def evaluate_goal(self, state) -> bool:
        """Evaluates if the current goal has been achieved"""
        pass
