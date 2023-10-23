# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy

from abc import ABC, abstractmethod
from random import randint
from typing import Callable, List, Tuple

import torch
from beanmachine.facebook.goal_inference.agent.boundedly_rational_agent import (
    BoundedRationalAgent,
)
from beanmachine.facebook.goal_inference.agent.observation_model import ObservationModel

from beanmachine.facebook.goal_inference.environment import State
from beanmachine.facebook.goal_inference.planner.planner import get_execution_path

from beanmachine.facebook.goal_inference.planner.stoch_astar import null_heuristic

from torch.distributions.categorical import Categorical


class DeviateTimeRejuvenateProposal:
    """Defines a probability distribution for proposing t*
    where t* is the time step where a proposed path deviates from the
    previous case.

    The (unnormalized) proability distribution is uniform (1.0) expect for the
    value where the P(Observation[i]|State[i]) decreases the most which has a
    value of max_prob

    The intuition is that the point with the greatest drop in P(Observation[i]|State[i])
    should be close to the point where this particle-agent deviates from the trajectory
    we are inferring

    Attributes/Arguments:

        max_prob: Weight of point with greatest decrease in P(Observation[i]|State[i])
    """

    def __init__(self, max_prob: float = 10.0):
        self.max_prob: float = max_prob

    def propose(
        self,
        agent: BoundedRationalAgent,
        observations: List[State],
        noise_model: ObservationModel,
    ) -> Tuple[int, torch.Tensor]:
        """Proposes a new time*, which will be used as the break point for the new agent-trajectory

        Arguments:
            agent: BoundedRationalAgent from whom a new path can be proposed
            observations: List of observations being inferred
            noise_model: Model for P(observation|state)

        Returns:
            time_deviate: Time step t* from which to propose new path
            log_prob: Log probability of proposed t*

        """
        proposal_distribution = self._get_prob(agent, observations, noise_model)
        time_deviate = Categorical(probs=proposal_distribution).sample().long()
        return time_deviate.item(), torch.log(proposal_distribution[time_deviate])

    def get_log_prob(
        self,
        agent: BoundedRationalAgent,
        observations: List[State],
        noise_model: ObservationModel,
        time_deviate: int,
    ) -> torch.Tensor:
        """Returns the log probability of a t* from the proposal distribution

        Arguments:
            agent: BoundedRationalAgent from whom a new path can be proposed
            observations: List of observations being inferred
            noise_model: Model for P(observation|state)
            time_deviate: Time step t* from which to propose new path

        Returns:
            log_prob: Log probability of proposed t*

        """
        return torch.log(self._get_prob(agent, observations, noise_model)[time_deviate])

    def _get_prob(
        self,
        agent: BoundedRationalAgent,
        observations: List[State],
        noise_model: ObservationModel,
    ) -> torch.Tensor:
        """Get Normalized Proposal Distribution P(t*|observations,states)

        Arguments:
            agent: BoundedRationalAgent from whom a new path can be proposed
            observations: List of observations being inferred
            noise_model: Model for P(observation|state)

        Returns:

            distribution: Normalized P(t*|observations,states)

        """
        agent_path = get_execution_path(agent.curr_node)[0]
        probs = torch.zeros(len(agent_path))
        # Compute P(observation|state) at each time step
        for time_step in range(len(agent_path)):
            probs[time_step] = torch.exp(
                noise_model.get_log_prob(agent_path[time_step], observations[time_step])
            )
        probs = torch.nan_to_num(probs, nan=0.0, neginf=0.0)
        prob_differences = probs[1:] - probs[0:-1]
        distribution = torch.ones(len(agent_path) - 1)
        # Sets probability of maximum probability decrease
        distribution[torch.argmin(prob_differences)] = self.max_prob
        distribution = distribution / torch.sum(distribution)
        return distribution


class PathRejuvenateProposal:
    """Defines a probability distribution for proposing new plans and states that
    deviate from an original trajectory

    During planning, Nodes are biased by an additional weight associated with the
    probability of that path given the observations.

    """

    def propose(
        self,
        agent: BoundedRationalAgent,
        observations: List[State],
        proposal_time_deviate: int,
        noise_model: ObservationModel,
    ) -> Tuple[BoundedRationalAgent, torch.Tensor]:
        """Proposes an agent with a new trajectory and the log probability of that agent

        Arguments:
            agent: BoundedRationalAgent from whom a new path can be proposed
            observations: List of observations being inferred
            proposal_time_deviate: Time step t* from which to propose new path
            noise_model: Model for P(observation|state)

        Returns:

            proposed_agent: An Agent with a modified path
            proposed_prob: The log probability of that proposed path

        """
        proposed_agent = copy.copy(agent)
        proposed_agent.propose_path(observations, proposal_time_deviate, noise_model)
        return proposed_agent, self.get_log_prob(
            proposed_agent, observations, proposal_time_deviate, noise_model
        )

    def get_log_prob(
        self,
        agent: BoundedRationalAgent,
        observations: List[State],
        proposal_time_deviate: int,
        noise_model: ObservationModel,
    ) -> torch.Tensor:
        """Evaluates the log probability of the new agent trajectory

        Arguments:
            agent: BoundedRationalAgent from whom a new path can be proposed
            observations: List of observations being inferred
            proposal_time_deviate: Time step t* from which to propose new path
            noise_model: Model for P(observation|state)

        Returns:
            log_prob: The log probability of that proposed path

        """
        return agent.get_log_prob(observations, proposal_time_deviate, noise_model)


class HeuristicGoalProposal:
    """Defines a goal proposal distribution using a given heuristic
    the proposal probability of each goal is
    proportional to ~exp(-heuristic/scale)

    Attributes/Arguments:

        heuristic: The heuristic on which the distribution is based
        scale: Determines the sharpness of the proposal distribution
    """

    def __init__(self, heuristic: Callable = null_heuristic, scale: float = 10.0):
        self.heuristic = heuristic
        self.scale = scale

    def propose(
        self,
        agent: BoundedRationalAgent,
        possible_goals: List[Tuple[str, ...]],
    ) -> Tuple[Tuple[str, ...], torch.Tensor]:
        """Proposes an new goal for an agent

        Arguments:
            agent: Agent whose goal can be modified
            possible_goals: Goals the agent can have

        Returns:
            possible_goal: New goal for the agent
            log_prob: Log Probability of the proposed goal

        """
        proposal_distribution = self._get_goal_log_probs(agent, possible_goals)
        goal_index = Categorical(logits=proposal_distribution).sample().long().item()
        return possible_goals[goal_index], proposal_distribution[goal_index]

    def get_log_prob(
        self,
        agent: BoundedRationalAgent,
        possible_goals: List[Tuple[str, ...]],
        goal: Tuple[str, ...],
    ) -> torch.Tensor:
        """Evaluates the log probability of a goal

        Arguments:
            agent: Agent whose goal can be modified
            possible_goals: Goals the agent can have
            goal: Target goal to evaluate probability of

        Returns:
            log_prob: Log Probability of the target goal

        """
        proposal_distribution = self._get_goal_log_probs(agent, possible_goals)
        target_goal_id = possible_goals.index(goal)
        return proposal_distribution[target_goal_id]

    def _get_goal_log_probs(
        self,
        agent: BoundedRationalAgent,
        possible_goals: List[Tuple[str, ...]],
    ) -> torch.Tensor:
        """Computes Log Proposal Distribution P(goal|state)

        Arguments:
            agent: Agent whose goal can be modified
            possible_goals: Goals the agent can have

        Returns:
            log_probs: Normalized Log P(goal|state)

        """
        curr_node = agent.curr_node
        heuristic_values = []
        for goal in possible_goals:
            heuristic_values.append(self.heuristic(curr_node, goal))
        heuristic_values = torch.tensor(heuristic_values)
        log_probs = -heuristic_values / self.scale
        log_probs -= torch.logsumexp(log_probs, dim=0)
        return log_probs


class GoalPrior(ABC):
    """Defines a distribution for sampling the prior distribution of goals of particle-agents"""

    @abstractmethod
    def sample_prior(self, possible_goals: List[Tuple[str, ...]]) -> Tuple[str, ...]:
        """Samples from the possible goals according to the prior

        Arguments:
            possible_goals: The possible goals to sample from

        Returns:
            sample_goal: The goal that was sampled

        """
        pass


class UniformGoalPrior(GoalPrior):

    """Defines a Uniform distribution for sampling the prior distribution of goals
    of particle-agents"""

    def sample_prior(self, possible_goals: List[Tuple[str, ...]]) -> Tuple[str, ...]:
        """Uniformly samples from possible goals

        Arguments:
            possible_goals: Goals that can be sampled

        Returns:
            goal: A possible goal drawn from a uniform distribution

        """
        number_of_goals = len(possible_goals)
        return possible_goals[randint(0, number_of_goals - 1)]


class StratifiedGoalPrior(GoalPrior):
    """Defines a (Stratified) Uniform distribution for sampling the prior distribution
    of goals of particle-agents

    Example - 300 particles with 3 goals -> exactly 100 particles / goal

    Arguments:
        number_of_samples: Total number of samples to draw from prior

    Attributes:
        number_of_samples: Total number of samples to draw from prior
        current_sample: Identifier for the nth sample out of number_of_samples

    """

    def __init__(self, number_of_samples: int):
        self.number_of_samples: int = number_of_samples
        self.current_sample: int = 0

    def sample_prior(self, possible_goals: List[Tuple[str, ...]]) -> Tuple[str, ...]:
        """Samples from Stratified Prior

        Arguments:
            possible_goals: Goals that can be sampled

        Returns:
            goal: A possible goal drawn from a stratified distribution

        """
        if self.current_sample < self.number_of_samples:
            current_goal = int(
                self.current_sample * len(possible_goals) / self.number_of_samples
            )
            self.current_sample += 1
            return possible_goals[current_goal]
        else:
            raise (RuntimeError("Too many samples drawn from stratified prior"))
