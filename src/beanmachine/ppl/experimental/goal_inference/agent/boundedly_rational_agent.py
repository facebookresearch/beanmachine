# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import copy

from typing import List, Tuple

import torch
from beanmachine.facebook.goal_inference.agent.observation_model import ObservationModel

from beanmachine.facebook.goal_inference.environment import State
from beanmachine.facebook.goal_inference.planner.planner import (
    get_execution_path,
    Plan,
    Planner,
    StateNode,
)

from beanmachine.facebook.goal_inference.planner.stoch_astar import (
    StochasticAstarProposalPlanner,
)
from torch.distributions.negative_binomial import NegativeBinomial


class BoundedRationalAgent:

    """
    An Agent that makes and executes short term plans based on budgetary contrainsts.

    Arguments:

        planner: The strategy for determining short term plans
        initial_state: Initial state of the problem
        r: Parameter of negative binomial distribution. Determines the number of failures that must be exceeded
        p: Parameter of negative binomial distribution. Determines the success/failure rate
        max_steps: Maximum number of steps to execute

    Attributes:

        planner: The strategy for determining short term plans
        r: Parameter of negative binomial distribution. Determines the number of failures that must be exceeded
        p: Parameter of negative binomial distribution. Determines the success/failure rate
        max_steps: Maximum number of steps to execute
        action_step: Action step within the current plan
        agent_step: Total number of steps by the agent
        budget_dist(r,p): A negative binomial distribution: Samples the number of successful trials before r failures if the probability of success is p
        plan_history: The series of plans constructed by this agent
        plan_index: Index of curr_plan in plan_history
        initial_node: Graph node defining the environment at the first time step.
        curr_node: Graph node defining environment at current time step
        proposal_planner: Proposal planner for path rejuvenation
    """

    def __init__(
        self,
        planner: Planner,
        initial_state: State,
        r: int = 2,
        p: float = 0.95,
        max_steps: int = 1000,
    ):
        self.planner: Planner = planner
        self.r: int = r
        self.p: float = p
        self.max_steps: int = max_steps
        self.agent_step: int = 0
        self.action_step: int = 0
        self.budget_dist: NegativeBinomial = NegativeBinomial(r, p)
        self.plan_history: List[Plan] = []
        self.plan_index: int = -1
        self.initial_node: StateNode = StateNode(initial_state, None, [])
        self.curr_node: StateNode = self.initial_node
        self.proposal_planner: StochasticAstarProposalPlanner = (
            StochasticAstarProposalPlanner(self.planner)
        )

    @property
    def curr_plan(self) -> Plan:
        """Gets the current plan based on the plan_history and plan_index

        Returns:
            curr_plan: The current plan
        """
        if self.plan_index == -1:
            return Plan([], [], [], 0)
        else:
            return self.plan_history[self.plan_index]

    def execute_search(self) -> Tuple[Plan, bool]:
        """Evolves the system to a solution or until max_steps is reached. Uses short term plans.

        Returns:
            plan: The plan executed by the agent
            solved: Whether a solution to the problem was reached
        """

        while (
            not self.planner.domain.evaluate_goal(self.curr_node.state)
            and self.agent_step < self.max_steps
        ):
            self.replan()
            self.execute_plan()

        return (
            # pyre-fixme[19]: Expected 4 positional arguments.
            Plan(
                *get_execution_path(self.curr_node),
                self.planner.visited_nodes,
                self.max_steps,
            ),
            self.planner.domain.evaluate_goal(self.curr_node.state),
        )

    def execute_plan(self) -> None:
        """Executes a short term plan. Stops Early if states deviate from expectation"""
        while self.action_step < len(self.curr_plan.actions):
            self._execute_step()

    def replan(self) -> None:
        """Updates the current plan based on the current state"""
        new_plan, solved = self.planner.generate_plan(
            self.curr_node.state,
            int(self.budget_dist.sample((1,)).item()),
        )
        # Don't consider empty plans which can occur if a solution is already reached
        if len(new_plan.actions) > 0:
            self.action_step = 0
            self.plan_history.append(new_plan)
            self.plan_index += 1

    def _execute_step(self) -> None:
        """Executes a single action from the current plan"""
        if not self.is_next_step_deviating():
            new_node = StateNode(
                self.curr_plan.states[self.action_step + 1],
                executed=self.curr_plan.actions[self.action_step],
                parent_node=self.curr_node,
            )
            self.curr_node = new_node
            self.agent_step += 1
            self.action_step += 1

        else:
            # The new state can deviate from the expected plan if actions are not deterministic
            # In that case, we need a new plan before continuing (otherwise planned actions may be invalid)
            self.action_step = len(self.curr_plan.actions)

    def is_next_step_deviating(self) -> bool:
        """Determines if next state deviates from expected plan

        Returns:
            is_deviating: Whether the next state deviates from the expectation of the plan
        """
        act = self.curr_plan.actions[self.action_step]
        next_predicted_state = self.curr_plan.states[self.action_step + 1]
        new_state = self.planner.domain.execute(self.curr_node.state, *act)
        return not new_state == next_predicted_state

    def step(self, num_steps: int = 1) -> None:
        """Advances the agent by one time step

        Arguments:
            num_steps: The number of time steps to advance
        """
        for _step in range(num_steps):
            while (
                self.action_step >= len(self.curr_plan.actions)
                or self.is_next_step_deviating()
            ):
                if self.planner.domain.evaluate_goal(self.curr_node.state):
                    break
                self.replan()
            # Make sure that current plan is not empty and agent is not at end of plan
            # This condition can fail if a solution has been reached
            if self.curr_plan.actions and self.action_step < len(
                self.curr_plan.actions
            ):
                self._execute_step()

    def propose_path(
        self,
        observations: List[State],
        proposal_time_deviation: int,
        noise_model: ObservationModel,
    ) -> None:
        """Proposes a new path for the bounded-rational agent starting from proposal_time_deviation

        Arguments:
            observations: The observed path
            proposal_time_deviation: The time step to start proposing a new path
            noise_model: A model for P(observation | state)
        """
        # Set time of bounded-agent to where proposed path diverges from previous path
        self._set_to_timestep(proposal_time_deviation, clear=True)
        while (
            not self.planner.domain.evaluate_goal(self.curr_node.state)
            and self.agent_step < len(observations) - 1
        ):
            while (
                self.action_step >= len(self.curr_plan.actions)
                or self.is_next_step_deviating()
            ):
                if self.planner.domain.evaluate_goal(self.curr_node.state):
                    break
                # Propose the next plan
                self.propose_plan(observations, noise_model)
            # Make sure that current plan is not empty and agent is not at end of plan
            # This condition can fail if a solution has been reached
            if self.curr_plan.actions and self.action_step < len(
                self.curr_plan.actions
            ):
                self._execute_step()

    def propose_plan(
        self, observations: List[State], noise_model: ObservationModel
    ) -> None:
        """Proposes the next plan during path rejuvenation

        Arguments:
            observations: The observed path
            noise_model: A model for P(observation | state)

        """
        new_plan, solved = self.proposal_planner.propose(
            self.curr_node.state,
            observations,
            self.budget_dist.sample((1,)).item(),
            noise_model,
        )
        # Add this plan to the history of this agent
        if len(new_plan.actions) > 0:
            self.action_step = 0
            self.plan_history.append(new_plan)
            self.plan_index += 1

    def __copy__(self) -> BoundedRationalAgent:
        """Creates a shallow copy of a BoundedRationalAgent

        Returns:
            agent_copy: A shallow copy of the agent
        """
        agent_copy = BoundedRationalAgent(
            self.planner,
            self.initial_node.state,
            r=self.r,
            p=self.p,
            max_steps=self.max_steps,
        )

        agent_copy.agent_step = self.agent_step
        agent_copy.action_step = self.action_step
        agent_copy.plan_history = copy.copy(self.plan_history)
        agent_copy.plan_index = self.plan_index
        agent_copy.curr_node = self.curr_node
        return agent_copy

    def get_log_prob(
        self,
        observations: List[State],
        proposal_time_deviation: int,
        noise_model: ObservationModel,
    ) -> torch.Tensor:
        """Evaluates the log probability of a series of plans

        Arguments:
            observations: The observed path
            proposal_time_deviation: The time step to start proposing a new path
            noise_model: A model for P(observation | state)

        Returns:
            log_prob: The log probability of the proposed series of plans

        """
        # Plans before divergence are not changed
        time_step = 0
        self.plan_index = 0
        while time_step < proposal_time_deviation:
            time_step += len(self.curr_plan.actions)
            self.plan_index += 1
        log_prob = torch.tensor(0.0)
        # Compute contribution to proposal probability from each plan
        while self.plan_index < len(self.plan_history):
            # Computes probability of plan budget
            log_prob_plan_length = self.budget_dist.log_prob(
                torch.tensor(self.curr_plan.budget)
            )
            # Computes probability of specific planning process

            log_prob_plan_visit = self.proposal_planner.get_log_prob(
                self.curr_plan,
                observations[
                    self.agent_step : self.agent_step + len(self.curr_plan.states)
                ],
                noise_model,
            )
            log_prob += log_prob_plan_length
            log_prob += log_prob_plan_visit
            self.plan_index += 1
        self.plan_index = len(self.plan_history) - 1

        return log_prob

    def _set_to_timestep(self, time_step: int, clear: bool = True):
        """Set time of bounded-agent to where proposed path diverges from previous path

        Arguments:
            proposal_time_deviation: The time step to start proposing a new path
            clear: Whether to remove future plans when setting the time

        """
        # Start at initial conditions of bounded agent
        self.plan_index = 0
        self.agent_step = 0
        self.action_step = 0
        self.curr_node = self.initial_node
        # Follow execute previous plans until deviation time
        while self.agent_step < time_step:
            new_node = StateNode(
                self.curr_plan.states[self.action_step + 1],
                executed=self.curr_plan.actions[self.action_step],
                parent_node=self.curr_node,
            )
            self.curr_node = new_node
            self.agent_step += 1
            self.action_step += 1
            # When plan is finished switch to the next one
            if (
                self.action_step == len(self.curr_plan.actions)
                and self.agent_step != time_step
            ):
                self.action_step = 0
                self.plan_index += 1

        # Erase the remainder of the plans from the previous path
        # Erase when proposing NOT when evaluating probability of proposal
        if clear:
            self.plan_history = self.plan_history[: self.plan_index + 1]
