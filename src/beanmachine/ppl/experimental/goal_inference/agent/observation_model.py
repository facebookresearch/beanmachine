# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses
from abc import ABC, abstractmethod
from typing import List

import torch
from beanmachine.facebook.goal_inference.environment import State


class ObservationModel(ABC):
    """Defines a probability distribution for observing states"""

    @abstractmethod
    def apply_noise(self, state) -> State:
        """Applies noise to a state to obtain an observation

        Arguments:
            state: The state to apply noise to

        Returns:
            noisy_state: The state with noise applied

        """
        pass

    @abstractmethod
    def get_log_prob(self, state, observation) -> torch.Tensor:
        """Gets the log probability of a observation given a state

        Arguments:
            state: The reference state
            observation: The noisy observation

        Returns:
            log_prob: The log of P(observation|state)

        """
        pass


class DeterministicObservation(ObservationModel):
    """Defines an observation model with no noise"""

    def apply_noise(self, state: State) -> State:
        """Applies no noise to a state

        Arguments:
            state: The state to apply noise to

        Returns:
            state: The state with no noise applied

        """
        return dataclasses.replace(state)

    def get_log_prob(self, state: State, observation: State) -> torch.Tensor:
        """Gets the log probability of a observation given a state assuming P(observation | state) is a delta function

        Arguments:
            state: The reference state
            observation: The noisy observation

        Returns:
            log_prob: The log of P(observation|state)

        """
        if state == observation:
            return torch.tensor(0.0)
        return torch.tensor(-float("inf"))


def apply_noise_to_states(
    states: List[State], noise_model: ObservationModel
) -> List[State]:
    """Applies noise to a series of states, returning a set of observations

    Arguments:
        states: The series of states to apply noise to
        noise_model: The model defining the applied noise

    Returns:
        observations: A series of observations corresponding to states with applied noise

    """
    observations = []
    for state in states:
        observations.append(noise_model.apply_noise(state))
    return observations
