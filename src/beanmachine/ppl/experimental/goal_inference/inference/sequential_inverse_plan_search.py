# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import dataclasses

from typing import Any, Dict, List, Tuple

import torch

from beanmachine.facebook.goal_inference.agent.boundedly_rational_agent import (
    BoundedRationalAgent,
)

from beanmachine.facebook.goal_inference.agent.observation_model import (
    DeterministicObservation,
    ObservationModel,
)

from beanmachine.facebook.goal_inference.environment import State

from beanmachine.facebook.goal_inference.inference.kernels import (
    DeviateTimeRejuvenateProposal,
    HeuristicGoalProposal,
    PathRejuvenateProposal,
    StratifiedGoalPrior,
)

from beanmachine.facebook.goal_inference.planner.planner import (
    get_execution_path,
    Planner,
)
from beanmachine.facebook.goal_inference.planner.stoch_astar import (
    null_heuristic,
    StochasticAstarPlanner,
)
from torch.distributions.categorical import Categorical


EPS = 10e-20

DEFAULT_NOISE_MODEL = DeterministicObservation()


class SIPS:

    """
    Peforms Sequential Inverse Planning Search algorithm for goal inference
    Full details can be found here: https://papers.nips.cc/paper/2020/file/df3aebc649f9e3b674eeb790a4da224e-Paper.pdf

    Summary of Algorithm:

    SIPS is a particle filtering scheme used to estimate the goal posterior P(g|obs) from observations
    where observations are obtained from an agent that does not necessarily make optimal choices.

    "Particles" are each associated with an agent trajectory generated assuming a particular goal.
    The particle-trajectory set is initialized with a uniform prior over possible goals.
    At each time step, each particle-trajectory is advanced one time step using a model agent.
    Each particle-trajectory receives a weight given the similarity to the target observations.
    Intuition: particle-trajectories with similar paths to the target observations are more likely
    to correspond to the target goal and thus receive higher weight.

    The posterior P(g|obs) at each time step can be obtained by summing the normalized weights associated
    with goal g.

    In order to ensure a good particle-trajectory distribution, if the effective sample size of the
    distribution decreases below a threshold, particles are resampled using multinomial sampling.

    In addition, the goals/trajectories of the particle-trajectories can be changed through goal/path rejuventation.
    This process involves a Metroplis Hastings steps with a proposal defined by the user

    Arguments:

        planner: The short-term strategy for particle-trajectories
        noise_model: Defines a probability distribution for observing states
        agent_args: Arguments for the boundedly-rational agents used for particle-trajectories

    Attributes:

        planner: The short-term strategy for particle-trajectories
        agent_args: Arguments for the boundedly-rational agents used for particle-trajectories
        noise_model: Defines a probability distribution for observing states
        path_proposal: Proposal that defines distribution for rejuvenating paths
        time_proposal: Proposal that defines distribution for t*, time step where to start path rejuvenation
        goal_prior: function that initialzes the goals of the particle-agents based on a prior
        goal_proposal: Proposal for proposing changes to the goal state

    """

    def __init__(
        self,
        planner: Planner,
        noise_model: ObservationModel = DEFAULT_NOISE_MODEL,
        **kwargs: Dict[str, Any],
    ):
        self.planner: Planner = planner
        self.agent_args: Dict[str, Any] = kwargs
        # Fall back on deterministic observations if no noise model is provided
        self.noise_model: ObservationModel = noise_model
        self.path_proposal: PathRejuvenateProposal = PathRejuvenateProposal()
        self.time_proposal: DeviateTimeRejuvenateProposal = (
            DeviateTimeRejuvenateProposal()
        )
        if isinstance(self.planner, StochasticAstarPlanner):
            self.goal_proposal: HeuristicGoalProposal = HeuristicGoalProposal(
                heuristic=self.planner.heuristic
            )
        else:
            self.goal_proposal: HeuristicGoalProposal = HeuristicGoalProposal(
                heuristic=null_heuristic
            )

    def _initialize_filter(
        self,
        num_particles: int,
        initial_state: State,
        possible_goals: List[Tuple[str, ...]],
    ) -> Tuple[torch.Tensor, List[BoundedRationalAgent]]:
        """Initializes the particle-agent distribution

        Arguments:
            num_particles: The total number of agents to initialize
            initial_state: The initial state of the problem
            possible_goals: The possible goals of the initial agents

        Returns:
            initialized_weights: Uniform weights for the agents
            agents: The initialized set of BoundedRational agents

        """
        agents = []
        goal_prior = StratifiedGoalPrior(num_particles)
        for _part_idx in range(num_particles):
            # Draw from P(g)
            sampled_goal = goal_prior.sample_prior(possible_goals)
            new_state = dataclasses.replace(initial_state, goal=sampled_goal)
            agents.append(
                BoundedRationalAgent(self.planner, new_state, **self.agent_args)
            )
        return torch.zeros(len(agents)), agents

    def infer(
        self,
        num_particles: int,
        initial_state: State,
        possible_goals: List[Tuple[str, ...]],
        observations: List[State],
        resample_threshold: float = 0.25,
        rejuvenate: bool = True,
        goal_rejuvenation_prob: float = 0.5,
    ) -> List[Dict[str, float]]:
        """Computes the normalized posterior P(g|obs) over time

        Arguments:
            num_particles: The number of particle-agents used in the filter
            initial_state: The initial state of the problem
            possible_goals: The possible goals that could be infered
            observations: The trajectory from which the goal will be inferred
            resample_threshold: Value for comparison with Effective_Sample_Size/num_particles.
                                If Effective_Sample_Size/num_particles falls below this value,
                                the particle distribution is resampled.
            rejuvenate: Whether to perform goal/path rejuvenation
            goal_rejuvenation_prob: Probability that goals are rejuvenated after resampling

        Returns:
            infered_goal: List of records of posterior. infered_goal[i] is the posterior at time_step i
        """
        # Initialize particle-trajectories. Goals are drawn from P(g)
        log_weights, agents = self._initialize_filter(
            num_particles, initial_state, possible_goals
        )
        infered_goal = []
        infered_goal.append(get_posterior(log_weights, agents, possible_goals))
        # Use observations to update particle-trajectory weights based on likelihood of observations
        for obs_step in range(1, len(observations)):
            agents, log_weights = self._update_filter(
                agents,
                log_weights,
                obs_step,
                observations[: obs_step + 1],
                initial_state,
                possible_goals,
                resample_threshold,
                rejuvenate,
                goal_rejuvenation_prob,
            )
            # Adjust weights for numerical stability

            log_weights -= torch.logsumexp(log_weights, dim=0)
            # Get posterior up until this time step
            infered_goal.append(get_posterior(log_weights, agents, possible_goals))

        return infered_goal

    def _update_filter(
        self,
        agents: List[BoundedRationalAgent],
        log_weights: torch.Tensor,
        obs_step: int,
        observations: List[State],
        initial_state: State,
        possible_goals: List[Tuple[str, ...]],
        resample_threshold: float,
        rejuvenate: bool,
        goal_rejuvenation_prob: float,
    ) -> Tuple[List[BoundedRationalAgent], torch.Tensor]:
        """Updates particle-agents with new observation

        Arguments:
            agents: Current set of BoundedRational Agents
            log_weights: Current agent weights
            obs_step: Current time_step of inference
            observations: Trajectory from which the goal will be inferred
            initial_state: Initial state of the problem
            possible_goals: The possible goals that can be infered
            resample_threshold: Value for comparison with Effective_Sample_Size/num_particles.
                                If Effective_Sample_Size/num_particles falls below this value,
                                the particle distribution is resampled.
            rejuvenate: Whether to perform goal/path rejuvenation
            goal_rejuvenation_prob: Probability that goals are rejuvenated after resampling

        Returns:
            agents: Updated BoundedRational Agents
            log_weights: Updated agent weights

        """
        # Test the effective sample size - resample if too low
        if (
            self._effective_sample_size(log_weights) / log_weights.shape[0]
            < resample_threshold
        ):
            log_weights, agent_idx = self._residual_resample(log_weights)

            for i in range(len(agents)):
                new_agent = copy.copy(agents[agent_idx[i]])
                if rejuvenate:
                    agents[i] = self._rejuvenate(
                        new_agent,
                        initial_state,
                        possible_goals,
                        observations,
                        goal_rejuvenation_prob,
                    )
                else:
                    agents[i] = new_agent

        for part_idx in range(len(log_weights)):
            # Update particle-trajectory using model agent
            agents[part_idx].step()
            # Update weight of this particle-trajectory P({observations}|{states}) with P(observation_i|state_i)
            log_weights[part_idx] += self.noise_model.get_log_prob(
                agents[part_idx].curr_node.state, observations[obs_step]
            )
        return agents, log_weights

    def _effective_sample_size(self, log_weights: torch.Tensor) -> torch.Tensor:
        """Computes the effective sample size of the current particles

        Arguments:
            log_weights: Current weights of the agents

        Returns:
            eff_sample_size: Effective sample size of the current distribution of agents

        """
        part_weights = get_real_weights(log_weights)
        sum_weights_sqrd = torch.square(torch.sum(part_weights))
        sum_sqrd_weights = torch.sum(torch.square(part_weights))
        eff_sample_size = sum_weights_sqrd / sum_sqrd_weights
        return eff_sample_size

    def _multinomial_resample(
        self, log_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resamples the particle distribution with multinomial resampling.

        New Agents are drawn from previous distribution in proportion to the weights

        Arguments:
            log_weights: Current weights of the agents

        Returns:
            uniform_weights: New Uniform weights
            sampled_indexes: Indexes of sampled agents
        """
        return (
            torch.zeros(log_weights.shape),
            Categorical(logits=log_weights).sample_n(log_weights.shape[0]).long(),
        )

    def _residual_resample(
        self, log_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resamples the particle distribution with residual resampling.

        Let the number of agents be N

        Previous Agents are sampled at least int(weight*N) times

        Remaining Agents are drawn in proportion to residual weights
        residual weight = [weight*N - int(weight*N)]

        Arguments:
            log_weights: Current weights of the agents

        Returns:
            uniform_weights: New Uniform weights
            sampled_indexes: Indexes of sampled agents
        """
        part_weights = get_real_weights(log_weights)
        N = part_weights.shape[0]
        num_copies = (N * part_weights).int()
        # Ensure each particle-agent is sampled at least int(N*part_weight) times
        agent_idx = torch.repeat_interleave(torch.arange(N), num_copies)
        k = torch.sum(num_copies)
        # Sample remaining agents based on decimal part of N*part_weight
        residual = N * part_weights - num_copies
        if torch.sum(residual) != 0.0 and N - k > 0:
            residual /= torch.sum(residual)
            agent_idx = torch.cat(
                (agent_idx, Categorical(probs=residual).sample_n(N - k)), dim=0
            )
        return torch.zeros(part_weights.shape), agent_idx.long()

    def _rejuvenate(
        self,
        agent: BoundedRationalAgent,
        initial_state: State,
        possible_goals: List[Tuple[str, ...]],
        observations: List[State],
        goal_rejuvenation_prob: float,
    ) -> BoundedRationalAgent:
        """Rejuvenates Agents by changing paths / goals

        Arguments:
            agent: The agent to modify
            initial_state: The initial state of the problem
            possible_goals: The possible goals to infer
            observations: Trajectory from which the goal will be inferred
            goal_rejuvenation_prob: Probability that goals are rejuvenated after resampling

        Returns:
            agent: The modified agent is return if the MC move is accepted. Otherwise the initial agent is returned.

        """
        # Determine whether to rejuventate goals or paths
        if torch.bernoulli(torch.tensor(goal_rejuvenation_prob)):
            alpha, proposed_agent = self._goal_rejuvenation(
                agent, initial_state, possible_goals
            )
        else:
            alpha, proposed_agent = self._path_rejuvenation(
                agent, initial_state, observations
            )
        # Evaluate Likelihood of path given observations P(states|Observations)
        old_path_log_prob = self._get_path_log_prob(agent, observations)
        proposed_path_log_prob = self._get_path_log_prob(proposed_agent, observations)
        # Accept/Reject Step
        alpha += proposed_path_log_prob - old_path_log_prob
        if torch.log(torch.rand(1)) < alpha:
            return proposed_agent
        else:
            return agent

    def _goal_rejuvenation(
        self,
        agent: BoundedRationalAgent,
        initial_state: State,
        possible_goals: List[Tuple[str, ...]],
    ) -> Tuple[torch.Tensor, BoundedRationalAgent]:
        """Rejuventates the goal of a particle-agent

        Proposes a new goal for a boundedrational agent
        The strategy for the proposal is determined by self.goal_proposal

        Currently, by default goals are

        Arguments:
            agent: The BoundedRational Agent to modify
            initial_state: The initial state of the problem
            possible_goals: The possible goals to infer

        Returns:
            log_proposal_prob: The proposal portion of the acceptance/rejection log probability
            proposed_agent: The modified agent
        """

        # Get proposed goal and log probability of goal
        proposed_goal, proposal_log_prob_new_goal = self.goal_proposal.propose(
            agent, possible_goals
        )
        # Get log probability of old goal
        proposal_log_prob_old_goal = self.goal_proposal.get_log_prob(
            agent, possible_goals, agent.curr_node.state.goal
        )
        # Propose new agent with sampled goal
        proposed_agent = BoundedRationalAgent(
            self.planner,
            dataclasses.replace(initial_state, goal=proposed_goal),
            **self.agent_args,
        )
        # Advance timestep of proposed agent to same time as previous agent
        proposed_agent.step(num_steps=agent.agent_step)
        return proposal_log_prob_old_goal - proposal_log_prob_new_goal, proposed_agent

    def _path_rejuvenation(
        self,
        agent: BoundedRationalAgent,
        initial_state: State,
        observations: List[State],
    ) -> Tuple[torch.Tensor, BoundedRationalAgent]:
        """Rejuventates the path of a paricle-agent

        (1) Proposes a time t* from which a new path can be proposed
        (2) Proposes a new path following time t*

        Arguments:
            agent: The BoundedRational Agent to modify
            initial_state: The initial state of the problem
            observations: Trajectory from which the goal will be inferred

        Returns:
            alpha: The proposal portion of the acceptance/rejection log probability
            proposed_agent: The modified agent

        """

        # Get t* and P(t*|state,observations)
        (
            proposal_time_deviate,
            proposal_log_prob_old_time_deviate,
        ) = self.time_proposal.propose(agent, observations, self.noise_model)

        # Get Proposed Agent  and P(Agent|observations,t*)
        proposed_agent, proposal_log_prob_new_path = self.path_proposal.propose(
            agent, observations, proposal_time_deviate, self.noise_model
        )

        # Get P(Old_Agent|observations,t*)
        proposal_log_prob_old_path = self.path_proposal.get_log_prob(
            agent, observations, proposal_time_deviate, self.noise_model
        )

        # Get P(t*|new_states,observations)
        proposal_log_prob_new_time_deviate = self.time_proposal.get_log_prob(
            proposed_agent, observations, self.noise_model, proposal_time_deviate
        )

        alpha = proposal_log_prob_old_path - proposal_log_prob_new_path
        alpha += proposal_log_prob_new_time_deviate - proposal_log_prob_old_time_deviate

        return alpha, proposed_agent

    def _get_path_log_prob(
        self, agent: BoundedRationalAgent, observations: List[State]
    ) -> torch.Tensor:
        """Computes Log P(observations|states) for a trajectory

        Arguments:
            agent: The agent to compare with the observations
            observations: Trajectory from which the goal will be inferred

        Returns:
            log_prob: Log P(observations|states)

        """
        log_prob = torch.tensor(0.0)
        agent_path = get_execution_path(agent.curr_node)[0]
        # Evaluates Log P(observations|states) = Log P(observation_1|State_1)+ Log P(observation_2|State_2)...
        for time_step in range(len(observations)):
            ### If the agent path has reached the goal just repeat the final state
            if time_step == len(agent_path):
                agent_path.append(agent_path[-1])
            # Computes Log P(observation_i|State_i)
            log_prob += self.noise_model.get_log_prob(
                agent_path[time_step], observations[time_step]
            )
        return log_prob


def get_real_weights(log_weights: torch.Tensor) -> torch.Tensor:
    """Computes weights from log weights in a numerically stable way

    Arguments:
        log_weights: Weights to convert to non-log scale

    Return:
        part_weights: Weights in non-log scale

    """
    log_weights -= torch.logsumexp(log_weights, dim=0)
    part_weights = torch.exp(log_weights)
    part_weights = torch.nan_to_num(part_weights, nan=0.0, neginf=0.0)
    return part_weights


def get_posterior(
    log_weights: torch.Tensor,
    agents: List[BoundedRationalAgent],
    possible_goals: List[Tuple[str, ...]],
) -> Dict[Tuple[str, ...], float]:
    """Computes the normalized posterior P(g|obs) at one time step

    Arguments:
        log_weights: Current weights of the agents
        agents: The current set of BoundedRational agents
        possible_goals: The possible goals to infer
    Returns:
        curr_infer: posterior at the current time_step
    """
    part_weights = get_real_weights(log_weights)
    curr_infer = {}
    for goal in possible_goals:
        curr_infer[goal] = 0.0
    norm = 0
    # Sum all weights associated with each goal
    for part_idx in range(len(agents)):
        norm += part_weights[part_idx].item()
        curr_infer[agents[part_idx].curr_node.state.goal] += part_weights[
            part_idx
        ].item()
    if norm == 0:
        return curr_infer

    # Normalize the posterior
    for goal_id in curr_infer:
        curr_infer[goal_id] /= norm

    return curr_infer
