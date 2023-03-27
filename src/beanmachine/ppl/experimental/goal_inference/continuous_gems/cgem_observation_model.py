# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import dataclasses

import torch
from beanmachine.facebook.goal_inference.agent.observation_model import ObservationModel
from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemState,
)
from beanmachine.facebook.goal_inference.environment import State
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal


class CGemObservation(ObservationModel):
    """Defines a probability distribution for observing states for a continuous gems problem.

    Agent positions (x,y) and angles are modified with Gaussian noise
    Held gems are modified with bernoulli noise

    Arguments/Attributes:
        position_noise: Scale of the noise applied when observing positions
        bernoulli_noise: Chance of gem being flipped from held->unheld or unheld -> held
        angle_noise: Scale of the noise applied when observing angles
    """

    def __init__(
        self,
        position_noise: float = 0.5,
        bernoulli_noise: float = 0.05,
        angle_noise=120.0,
    ):
        self.position_noise: Normal = Normal(0.0, position_noise)
        self.bernoulli_noise: Bernoulli = Bernoulli(bernoulli_noise)
        self.angle_noise: Normal = Normal(0.0, angle_noise)

    def apply_noise(self, state: CGemState) -> State:
        """Applies noise to a state to obtain an observation

        Arguments:
            state: The state to apply noise to

        Returns:
            noisy_state: The state with noise applied

        """

        new_at = copy.copy(state.at)
        new_has = copy.copy(state.has)

        # Apply Noise to gems
        for gem_id in state.gems:
            if gem_id in new_at:
                # Chance that gem at (x,y) is not observed
                if self.bernoulli_noise.sample():
                    new_at.pop(gem_id)

            else:
                # Chance that held gem is not observed
                if self.bernoulli_noise.sample():
                    new_has.pop(gem_id)

        # Apply Gaussian noise to positions / angles
        return dataclasses.replace(
            state,
            has=new_has,
            at=new_at,
            x=state.x + self.position_noise.sample().item(),
            y=state.y + self.position_noise.sample().item(),
            angle=(state.angle + self.angle_noise.sample().item()) % 360.0,
        )

    def get_log_prob(self, state: CGemState, observation: CGemState) -> torch.Tensor:
        """Gets the log probability of a observation given a state

        Arguments:
            state: The reference state
            observation: The noisy observation

        Returns:
            log_prob: The log of P(observation|state)

        """

        log_prob_gems = 0.0
        # Evaluate Probability of Item positions
        for gem_id in state.gems:
            # If gem is not flipped
            if (gem_id in state.has and gem_id in observation.has) or (
                gem_id in state.at and gem_id in observation.at
            ):
                log_prob_gems += self.bernoulli_noise.log_prob(torch.tensor(0.0))
            # If gem is flipped
            else:
                log_prob_gems += self.bernoulli_noise.log_prob(torch.tensor(1.0))

        # Evaluate probability of positions
        log_prob_x = self.position_noise.log_prob(torch.tensor(state.x - observation.x))
        log_prob_y = self.position_noise.log_prob(torch.tensor(state.y - observation.y))

        # Evaluate probability of angles
        difference = torch.tensor(state.angle - observation.angle)
        if difference > 180.0:
            difference = 360.0 - difference
        elif difference < -180.0:
            difference = -(difference + 360.0)

        log_prob_angle = self.angle_noise.log_prob(difference)

        return log_prob_x + log_prob_y + log_prob_angle + log_prob_gems
