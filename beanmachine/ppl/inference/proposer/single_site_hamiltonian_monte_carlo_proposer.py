# Copyright (c) Facebook, Inc. and its affiliates
import logging
from typing import Dict, Tuple

import torch
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    compute_first_gradient,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import Variable, World
from torch import Tensor, tensor


LOGGER_WARNING = logging.getLogger("beanmachine.warning")


class SingleSiteHamiltonianMonteCarloProposer(SingleSiteAncestralProposer):
    def __init__(self, path_length: float, step_size: float = 0.01):
        super().__init__()
        # mass matrix parameters
        self.mass_matrix_initialized = False
        self.sample_mean = None
        self.co_moment = None
        self.covariance = None
        self.no_cov_iterations = 75

        # step size parameters
        self.initialized = False
        self.gamma = 0.05
        self.t = 10
        self.kappa = 0.75
        self.optimal_acceptance_prob = 0.65
        self.step_size = tensor(step_size)
        self.path_length = tensor(path_length)
        self.mu = torch.log(10.0 * self.step_size)
        self.best_step_size = step_size
        self.closeness = 0

        # run-time fallback to MH
        self.runtime_error = False
        self.max_num_steps = 10000

    def _compute_new_step_acceptance_probability(
        self,
        node: RVIdentifier,
        node_var: Variable,
        world: World,
        q_unconstrained: Tensor,
        p: Tensor,
        original_grad: Tensor,
    ) -> Tensor:
        """
        Computes the acceptance probability for a node based on new value q_new,
        original momentum p_old, and new momentum p_new

        :param node: the node that we proposing a value for
        :param node_var: the Variable object associated with node
        :param world: the world where the node exists
        :param q_unconstrained: the original value
        :param p: the sampled momentum
        :param original_grad: the gradient at q_unconstrained
        :returns: the acceptance probability given the new q and p.
        """
        # make one leapfrog step
        p_new = p - self.step_size * original_grad / 2
        q_new = q_unconstrained + self.step_size * p_new
        is_valid, step_grad = self._compute_potential_energy_gradient(
            node, world, q_new
        )
        if not is_valid:
            return tensor(1e-10, dtype=q_unconstrained.dtype)
        p_new = p_new - self.step_size * step_grad / 2

        # compute acceptance probability
        current_K = self._compute_kinetic_energy(p)
        proposed_K = self._compute_kinetic_energy(p_new)
        q = node_var.transform_from_unconstrained_to_constrained(q_new)
        children_log_update, _, node_log_update, _ = world.propose_change(
            node, q, False
        )
        world.reset_diff()
        acceptance_log_prob = (
            children_log_update + node_log_update + current_K - proposed_K
        )
        return torch.exp(acceptance_log_prob)

    def _find_reasonable_step_size(self, node: RVIdentifier, world: World):
        """
        Calculated using Algorithm 4: Heurisitc for choosing an initial step size
        http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

        :param node: the node that we are finding the step size for
        :param world: the world where the node exists
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        if node_var.value is None:
            raise ValueError(f"{node} has no value")

        # initialize momentum and kinetic energy
        q_unconstrained = node_var.unconstrained_value
        p = torch.randn(q_unconstrained.shape)

        # take one leapfrog step with step size = 1.0
        self.step_size = tensor(1.0, dtype=q_unconstrained.dtype)
        is_valid, original_grad = self._compute_potential_energy_gradient(
            node, world, q_unconstrained
        )
        if not is_valid:
            return
        acceptance_prob = self._compute_new_step_acceptance_probability(
            node, node_var, world, q_unconstrained, p, original_grad
        )
        if acceptance_prob > tensor(0.5):
            a = 1
        else:
            a = -1
        step_size_multiplier = torch.pow(tensor(2.0), a)
        threshold = torch.pow(tensor(2.0), -a)

        max_iterations = 100
        for _ in range(max_iterations):
            # half or double step size
            self.step_size = step_size_multiplier * self.step_size
            acceptance_prob = self._compute_new_step_acceptance_probability(
                node, node_var, world, q_unconstrained, p, original_grad
            )
            if torch.pow(acceptance_prob, a) < threshold:
                # stop if the acceptance probability crosses the threshold
                break

    def _adapt_step_size(self, node, world, acceptance_probability, iteration_number):
        """
        Adapt the step size based on the acceptance probabity of the previous sample
        Calculated using Algorithm 5: HMC with Dual Averaging
        http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param acceptance_probability: the acceptance probability of the previous move.
        :param iteration_number: The current iteration of inference
        """
        iteration_number = tensor(iteration_number, dtype=acceptance_probability.dtype)
        closeness_frac = 1 / (iteration_number + self.t)
        self.closeness = ((1 - closeness_frac) * self.closeness) + (
            closeness_frac * (self.optimal_acceptance_prob - acceptance_probability)
        )

        log_step_size = self.mu - (
            (torch.sqrt(iteration_number) / self.gamma) * self.closeness
        )

        step_frac = torch.pow(iteration_number, -self.kappa)
        log_best_step_size = (step_frac * log_step_size) + (
            (1 - step_frac) * torch.log(self.best_step_size)
        )

        self.step_size = torch.exp(log_step_size)
        self.best_step_size = torch.exp(log_best_step_size)

    def _adapt_mass_matrix(self, iteration: int, sample: Tensor):
        """
        The mass matrix is approximated using the covariance of the samples
        Calculated using online covariance algorithm
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

        :param iteration: number of samples that have been used for covariance
        :param sample: the previous sample returned from inference
        """
        sample_vector = torch.reshape(sample, (-1,))

        if not self.mass_matrix_initialized:
            self.mass_matrix_initialized = True
            self.sample_mean = torch.zeros(len(sample_vector), dtype=sample.dtype)
            self.co_moment = torch.zeros(
                len(sample_vector), len(sample_vector), dtype=sample.dtype
            )
        old_sample_mean = self.sample_mean
        self.sample_mean = (
            self.sample_mean + (sample_vector - self.sample_mean) / iteration
        )
        x_term = (sample_vector - old_sample_mean).unsqueeze(0).T
        y_term = (sample_vector - self.sample_mean).unsqueeze(0)
        self.co_moment = self.co_moment + torch.matmul(x_term, y_term)

        if iteration > 1:
            covariance = self.co_moment / (iteration - 1)
            # smoothing the covariance matrix
            delta = tensor(1e-3, dtype=sample.dtype)
            identity = torch.eye(len(sample_vector), dtype=sample.dtype)
            self.covariance = (delta * identity) + ((1 - delta) * covariance)

    def do_adaptation(
        self,
        node: RVIdentifier,
        world: World,
        acceptance_probability: Tensor,
        iteration_number: int,
        num_adaptive_samples: int,
        is_accepted: bool,
    ) -> None:
        """
        To be implemented by proposers that are capable of adaptation at
        the beginning of the chain.

        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param acceptance_probability: the acceptance probability of the previous move.
        :param iteration_number: The current iteration of inference
        :param num_adapt_steps: The number of inference iterations for adaptation.
        :param is_accepted: bool representing whether the new value was accepted.
        :returns: Nothing.
        """
        iteration_number = iteration_number + 1
        if iteration_number > num_adaptive_samples:
            self.step_size = self.best_step_size
            return

        if self.runtime_error:
            acceptance_probability = tensor(0.0, dtype=acceptance_probability.dtype)

        if not self.initialized:
            self._find_reasonable_step_size(node, world)
            self.mu = torch.log(10 * self.step_size)
            self.best_step_size = tensor(1.0, dtype=acceptance_probability.dtype)
            self.initialized = True

        node_var = world.variables_.get_node_raise_error(node)
        self._adapt_step_size(node, world, acceptance_probability, iteration_number)
        if iteration_number > self.no_cov_iterations:
            self._adapt_mass_matrix(
                iteration_number - self.no_cov_iterations, node_var.unconstrained_value
            )

    def _compute_kinetic_energy(self, p: Tensor) -> Tensor:
        """
        To be implemented by proposers that are capable of adaptation at
        the beginning of the chain.

        :param p: the momentum
        :returns: the kinetic energy calculated by 1/2 * p.T * cov * p
        """
        p_vector = torch.reshape(p, (-1,))
        if self.covariance is None:
            self.covariance = torch.eye(len(p_vector), dtype=p.dtype)
        return torch.matmul(p_vector.T, torch.matmul(self.covariance, p_vector)) / 2

    def _compute_potential_energy_gradient(self, node, world, q_unconstrained):
        """
        Compute the gradient of the likelihood w.r.t the new unconstrained value

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param q_unconstrained: the proposed unconstrained value for the node
        """
        score = world.propose_change_unconstrained_value(
            node, q_unconstrained, allow_graph_update=False
        )[3]
        node_var = world.get_node_in_world(node, False)
        is_valid, grad_U = compute_first_gradient(
            -score, node_var.unconstrained_value, retain_graph=True
        )
        if not is_valid:
            LOGGER_WARNING.warning(
                "Gradient is invalid at node {n}: {nv}.\n".format(
                    n=str(node), nv=str(node_var)
                )
            )
        world.reset_diff()
        return is_valid, grad_U

    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the difference in kinetic
        energy between the start and the end value
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        if node_var.value is None:
            raise ValueError(f"{node} has no value")
        q_unconstrained = node_var.unconstrained_value

        # initialize momentum
        p = torch.randn(q_unconstrained.shape)
        current_K = self._compute_kinetic_energy(p)

        is_valid, grad_U = self._compute_potential_energy_gradient(
            node, world, q_unconstrained
        )
        if not is_valid:
            self.runtime_error = True
            zero_grad(q_unconstrained)
            LOGGER_WARNING.warning(
                "Node {n} has invalid proposal solution. ".format(n=node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().propose(node, world)

        # take a half-step for momentum
        p = p - self.step_size * grad_U / 2

        ideal_num_steps = int(torch.ceil(self.path_length / self.step_size))
        num_steps = min(ideal_num_steps, self.max_num_steps)
        # leapfrog steps
        for i in range(num_steps):
            q_unconstrained = q_unconstrained.detach()
            q_unconstrained = q_unconstrained + self.step_size * p
            is_valid, grad_U = self._compute_potential_energy_gradient(
                node, world, q_unconstrained
            )
            if not is_valid:
                self.runtime_error = True
                zero_grad(q_unconstrained)
                LOGGER_WARNING.warning(
                    "Node {n} has invalid proposal solution. ".format(n=node)
                    + "Proposer falls back to SingleSiteAncestralProposer.\n"
                )
                return super().propose(node, world)

            if i < num_steps - 1:
                p = p - self.step_size * grad_U

        # final half-step for momentum
        p = p - self.step_size * grad_U / 2

        zero_grad(q_unconstrained)
        proposed_K = self._compute_kinetic_energy(p)
        q = node_var.transform_from_unconstrained_to_constrained(q_unconstrained)
        self.runtime_error = False
        return q.detach(), current_K - proposed_K, {}

    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :returns: the log probability of proposing the old value from this new world.
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        if node_var.value is None:
            raise ValueError(f"{node} has no value")
        return torch.tensor(0.0, dtype=node_var.value.dtype)
