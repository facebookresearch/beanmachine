# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import dataclasses

import torch
from beanmachine.facebook.goal_inference.agent.observation_model import ObservationModel
from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import DKGState
from beanmachine.facebook.goal_inference.environment import State
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal


class DKGObservation(ObservationModel):
    """Defines a probability distribution for observing states for Doors, Keys, and Gems problem.
    Agent positions (x,y) are modified with Gaussian noise
    Presence of held Items, Doors, and Items on the ground is modified with bernoulli noise

    Arguments/Attributes:
        gaussian_noise: Scale of the noise applied when observing positions
        bernoulli_noise: Chance of a held Item, Door, or Item on the ground not being observed

    """

    def __init__(self, gaussian_noise: float = 0.25, bernoulli_noise: float = 0.05):
        self.gaussian_noise: Normal = Normal(0.0, gaussian_noise)
        self.bernoulli_noise: Bernoulli = Bernoulli(bernoulli_noise)

    def apply_noise(self, state: DKGState) -> State:
        """Applies noise to a state to obtain an observation

        Arguments:
            state: The state to apply noise to

        Returns:
            noisy_state: The state with noise applied
        """
        new_at = copy.copy(state.at)
        new_has = copy.copy(state.has)
        new_doors = copy.copy(state.doors)

        # Apply Noise to items
        for item_id in state.items:
            if item_id in new_at:
                # Chance that item at (x,y) is not observed
                if self.bernoulli_noise.sample():
                    new_at.pop(item_id)

            else:
                # Chance that held item is not observed
                if self.bernoulli_noise.sample():
                    new_has.pop(item_id)

        # Apply Noise to doors
        for doorloc in state.doors:
            # Change door is not observed
            if self.bernoulli_noise.sample():
                new_doors.remove(doorloc)

        # Apply Gaussian noise to positions
        return dataclasses.replace(
            state,
            at=new_at,
            has=new_has,
            doors=new_doors,
            x=state.x + self.gaussian_noise.sample().item(),
            y=state.y + self.gaussian_noise.sample().item(),
        )

    def get_log_prob(self, state: DKGState, observation: DKGState) -> torch.Tensor:
        """Gets the log probability of a observation given a state

        Arguments:
            state: The reference state
            observation: The noisy observation

        Returns:
            log_prob: The log of P(observation|state)

        """
        log_prob_items = 0.0
        # Evaluate Probability of Item positions
        for item_id in state.items:
            # If Item is not flipped
            if (item_id in state.has and item_id in observation.has) or (
                item_id in state.at and item_id in observation.at
            ):
                log_prob_items += self.bernoulli_noise.log_prob(torch.tensor(0.0))
            # If Item is flipped
            else:
                log_prob_items += self.bernoulli_noise.log_prob(torch.tensor(1.0))

        # Evaluate Probability of door positions
        log_prob_doors = 0.0
        for doorloc in state.doors:
            # If Door is not flipped
            if doorloc in observation.doors:
                log_prob_doors += self.bernoulli_noise.log_prob(torch.tensor(0.0))
            # If Door is flipped
            else:
                log_prob_doors += self.bernoulli_noise.log_prob(torch.tensor(1.0))

        # Evaluate probability of positions
        log_prob_x = self.gaussian_noise.log_prob(torch.tensor(state.x - observation.x))
        log_prob_y = self.gaussian_noise.log_prob(torch.tensor(state.y - observation.y))
        return log_prob_x + log_prob_y + log_prob_items + log_prob_doors
