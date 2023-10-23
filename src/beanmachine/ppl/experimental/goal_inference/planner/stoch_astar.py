# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
from beanmachine.facebook.goal_inference.agent.observation_model import (
    DeterministicObservation,
    ObservationModel,
)

from beanmachine.facebook.goal_inference.environment import Domain, State
from beanmachine.facebook.goal_inference.planner.planner import (
    get_execution_path,
    Plan,
    Planner,
    StateNode,
)


class StochasticAstarPlanner(Planner):

    """
    Obtains a probabilistic A* solution to a problem given a heuristic.
    The next node to explore is determined probabilistic sampling based on priorities
    where priority = cost (to reach that node) + heuristic (estimating additional actions until goal)
    and weights are proportional to exp(-priority/noise)

    Arguments:
        domain: Domain that encodes the rules of the world
        heuristic: Heuristic for A* Algorithm
        noise: Controls randomness of sampling process

    Attributes:
        domain: Domain that encodes the rules of the world
        visited_nodes: Record of nodes visited during planning
        heuristic: Heuristic for A* Algorithm
        cost: Record of cost to reach visited States
        weights: Proportional to probability of sampling a state
        nodes: Current nodes that can be sampled
        noise: Controls randomness of sampling process

    """

    def __init__(self, domain: Domain, heuristic: Callable, noise: float):
        super().__init__(domain)
        self.heuristic: Callable = heuristic
        self.cost: Dict[State, int] = {}
        self.weights: Dict[State, float] = {}
        self.nodes: Dict[State, StateNode] = {}
        self.noise: float = noise

    def _sample(self) -> StateNode:
        """Samples from nodes with probabilities proportional to weights

        Returns:
            node: Sampled StateNode
        """
        weight_tensor = torch.tensor(list(self.weights.values()))
        sampled_index = torch.distributions.categorical.Categorical(
            logits=weight_tensor
        ).sample()
        sampled_state = list(self.weights.keys())[sampled_index]
        return self._sample_node(sampled_state)

    def _sample_node(self, state: State) -> StateNode:
        """Samples target node. Removes node and corresponding weight from planner

        Arguments:
            state: State being sampled
        Returns:
            state_node: StateNode corresponding to the sampled state
        """
        self.weights.pop(state)
        return self.nodes.pop(state)

    def is_empty(self) -> bool:
        """Determines whether the prioirty queue is empty

        Returns:
            is_empty: Whether the astar datastructure is empty
        """
        return not self.nodes

    def reset(self) -> None:
        """Clears the datatructures associated with this StochasticAstarPlanner"""
        self.cost = {}
        self.weights = {}
        self.nodes = {}
        self.visited_nodes = []

    def get_next_node(self) -> StateNode:
        """Get next StateNode to evaluate

        Returns:
            The next node to evaluate
        """
        curr_node = self._sample()
        return curr_node

    def add_node(self, new_node: StateNode) -> None:
        """Add a node to StochasticAstar datastructure

        Arguments:
            new_node: The node to be potentially explored
        """
        if new_node.parent_node is None:
            self.cost[new_node.state] = 0
            self.weights[new_node.state] = 0.0
            self.nodes[new_node.state] = new_node
        else:
            new_cost = self.cost[new_node.parent_node.state] + 1
            if new_node.state not in self.cost or new_cost < self.cost[new_node.state]:
                self.cost[new_node.state] = new_cost
                priority = new_cost + self.heuristic(new_node)
                # Weight is proportional to exp[-priority/noise]
                self.weights[new_node.state] = -priority / self.noise
                self.nodes[new_node.state] = new_node


class StochasticAstarProposalPlanner(StochasticAstarPlanner):

    """
    Proposes a new path using probabilistic A* algorithm.
    The next node to explore is determined probabilistic sampling based on priorities
    where priority = cost (to reach that node) + heuristic (estimating additional actions until goal)
    and weights are proportional to exp(-priority/noise)*P(next_observation|next_state)

    P(next_observation|next_state) is an additional bias to encourage paths near future observations

    Arguments:
        agent_planner: The planning algorithm for the BoundedRationalAgent

    Attributes:
        domain: Domain that encodes the rules of the world
        visited_nodes: Record of nodes visited during planning
        heuristic: Heuristic for A* Algorithm. If the agent_planner was not a StochasticA*Planner, the heuristic just returns 0.0 for any state
        noise: Controls randomness of sampling process
        cost: Record of cost to reach visited States
        weights: Proportional to probability of sampling a state
        nodes: Current nodes that can be sampled
        noise_model: A model for P(observation|state)
        observations: List of future observations
    """

    def __init__(self, agent_planner: Planner):
        self.heuristic = null_heuristic
        self.noise: float = 0.1
        if isinstance(agent_planner, StochasticAstarPlanner):
            self.heuristic: Callable = agent_planner.heuristic
            self.noise: float = agent_planner.noise
        super().__init__(agent_planner.domain, self.heuristic, self.noise)
        self.observations: List[State] = []
        self.noise_model: ObservationModel = DeterministicObservation()

    def _sample(self) -> StateNode:
        """Samples from nodes with probabilities proportional to weights

        Returns:
            node: Sampled StateNode
        """
        weight_tensor = self._get_next_state_weights()
        sampled_index = torch.distributions.categorical.Categorical(
            logits=weight_tensor
        ).sample()
        sampled_state = list(self.weights.keys())[sampled_index]
        return self._sample_node(sampled_state)

    def _get_next_state_weights(self) -> torch.Tensor:
        """Defines the biasing of the proposal distribution.
        Probability of a node being sampled is proportional to exp[-heuristic/noise] * P(next_observation|next_state)

        Returns:
            biased_weights: Unnormalized Log weights associeated with the probability of each node being sampled
        """
        biased_weights = torch.tensor(list(self.weights.values()))
        biased_weights = torch.nan_to_num(biased_weights, nan=-float("inf"))
        biased_weights = torch.nan_to_num(biased_weights)
        return biased_weights

    def add_node(self, new_node: StateNode) -> None:
        """Add a node to StochasticAstarProposal datastructure

        Arguments:
            new_node: The node to be potentially explored
        """
        if new_node.parent_node is None:
            self.cost[new_node.state] = 0
            self.weights[new_node.state] = 0.0
            self.nodes[new_node.state] = new_node
        else:
            new_cost = self.cost[new_node.parent_node.state] + 1
            if new_node.state not in self.cost or new_cost < self.cost[new_node.state]:
                self.cost[new_node.state] = new_cost
                priority = new_cost + self.heuristic(new_node)
                bias = 0.0
                curr_path = get_execution_path(new_node)[0]
                if len(curr_path) - 1 < len(self.observations):
                    bias = self.noise_model.get_log_prob(
                        curr_path[-1], self.observations[len(curr_path) - 1]
                    )
                # Naive weight is proportional to exp[-priority/noise]
                # Adjust priority based on P(next_observation|next_state)
                self.weights[new_node.state] = -priority / self.noise + bias
                self.nodes[new_node.state] = new_node

    def propose(
        self,
        state: State,
        observations: List[State],
        budget: int,
        noise_model: ObservationModel,
    ) -> Tuple[Plan, bool]:
        """Propose a plan using the biased proposal distribution

        Arguments:
            state: The starting state of the plan
            observations: List of future observations
            budget: Node search limit for the planning process
            noise_model: A model for P(observation|state)

        Returns:
            plan: A new partial plan to reach the goal
            solved: Whether the plan reached the goal
        """
        self.observations = observations
        self.noise_model = noise_model
        return self.generate_plan(state, budget)

    def get_log_prob(
        self, plan: Plan, observations: List[State], noise_model: ObservationModel
    ) -> torch.Tensor:
        """Get the log probability of a specific bounded-rational agent planning sequence

        Arguments:
            plan: The planning process to compute probability of
            observations: List of future observations
            noise_model: A model for P(observation|state)

        Returns:
            log_prob: Log probability of the plan
        """
        # Initialize planning
        self.reset()
        self.observations = observations
        self.noise_model = noise_model
        init_node = StateNode(plan.states[0], None, [])
        self.add_node(init_node)
        log_prob = torch.tensor(0.0)
        # Follow the planning decisions and update the probability of the
        # planning sequence with each decision
        for planning_step in range(len(plan.visited_nodes)):
            # Get the probability distribution at this stage of planning
            current_distribution = self._get_next_state_weights()
            current_distribution -= torch.logsumexp(current_distribution, dim=0)
            curr_visited_state = plan.visited_nodes[planning_step].state
            # Determine which node was sampled
            for node_idx in range(len(self.nodes)):
                curr_node_state = list(self.nodes.keys())[node_idx]
                if curr_node_state == curr_visited_state:
                    log_prob += current_distribution[node_idx]
                    curr_node = self._sample_node(curr_node_state)
                    # Add next possible nodes to replicate planning process
                    poss_actions = self.domain.get_possible_actions(curr_node.state)
                    for act in poss_actions:
                        next_state = self.domain.execute(curr_node.state, *act)
                        new_node = StateNode(
                            next_state, executed=act, parent_node=curr_node
                        )
                        self.add_node(new_node)
                    break

        return log_prob


def null_heuristic(state: State) -> float:
    """Default heuristic for StochasticAstarProposalPlanner
    Returns 0 cost for all goals for all states

    Arguments:
       state: State to predict additional cost to goal

    Returns:
       cost: =0.0 in all cases
    """
    return 0.0
