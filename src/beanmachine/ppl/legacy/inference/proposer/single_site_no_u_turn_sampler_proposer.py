# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Tuple, Union

import torch
import torch.distributions as dist
from beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils import (
    compute_first_gradient,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.world import Variable, World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor, tensor


class SingleSiteNoUTurnSamplerProposer(SingleSiteAncestralProposer):
    def __init__(self, use_dense_mass_matrix: bool = True):
        super().__init__()
        # NUTS parameters
        self.max_depth = 5
        self.delta_max = 1000
        self.alpha = 0.0
        self.n_alpha = 1.0
        self.ratio = 0.0

        # mass matrix parameters
        self.use_dense_mass_matrix = use_dense_mass_matrix
        if self.use_dense_mass_matrix:
            self.mass_matrix_initialized = False
            self.sample_mean = None
            self.co_moment = None
            self.start_cov = 75
            self.initial_window_size = 25
            self.window_size = self.initial_window_size
            self.covariance = None
            self.track_covariance = None
            self.l_inv = None
            self.covariance_diagonal_padding = 1e-6

        # step size parameters
        self.initialized = False
        self.gamma = 0.05
        self.t = 10
        self.kappa = 0.75
        self.optimal_acceptance_prob = 0.65
        self.step_size = 0.1
        self.mu = math.log(10.0 * self.step_size)
        self.best_step_size = self.step_size
        self.closeness = 0
        self.max_initial_iterations = 100

    def _compute_new_step_acceptance_probability(
        self,
        node: RVIdentifier,
        node_var: Variable,
        world: World,
        theta: Tensor,
        r: Tensor,
        step_size: Union[float, Tensor],
    ) -> Tensor:
        """
        Computes the acceptance probability for a step given the
        current theta and r

        :param node: the node that we proposing a value for
        :param node_var: the Variable object associated with node
        :param world: the world where the node exists
        :param theta: the original theta
        :param r: the sampled momentum
        :param step_size: the leapfrog step size
        :returns: the acceptance probability given the new theta and r
        """
        # make one leapfrog step
        is_valid, theta1, r1 = self._leapfrog_step(node, world, theta, r, step_size)
        if not is_valid:
            return tensor(1e-10, dtype=theta.dtype)

        # compute acceptance probability
        current_K = self._compute_kinetic_energy(r)
        proposed_K = self._compute_kinetic_energy(r1)
        (
            children_log_update,
            _,
            node_log_update,
            _,
        ) = world.propose_change_transformed_value(node, theta1, False)
        world.reset_diff()
        acceptance_log_prob = (
            children_log_update + node_log_update + current_K - proposed_K
        )
        return torch.exp(
            torch.min(
                tensor(
                    0.0,
                    dtype=acceptance_log_prob.dtype,
                    device=acceptance_log_prob.device,
                ),
                acceptance_log_prob.detach(),
            )
        )

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
        theta = node_var.transformed_value
        r = torch.randn(theta.shape, dtype=theta.dtype, device=theta.device)

        step_size = 1.0
        # take one leapfrog step with step size = 1.0
        acceptance_prob = self._compute_new_step_acceptance_probability(
            node, node_var, world, theta, r, step_size
        )
        if acceptance_prob > tensor(
            0.5, dtype=acceptance_prob.dtype, device=acceptance_prob.device
        ):
            a = 1
        else:
            a = -1
        step_size_multiplier = 2.0 ** a
        threshold = 2.0 ** (-a)
        epsilon = tensor(
            1e-10, dtype=acceptance_prob.dtype, device=acceptance_prob.device
        )

        for _ in range(self.max_initial_iterations):
            # half or double step size
            step_size = step_size_multiplier * step_size
            acceptance_prob = self._compute_new_step_acceptance_probability(
                node, node_var, world, theta, r, step_size
            )
            # Set lower bound of acceptance probability to epsilon to prevent running
            # into numerical issue in computing 0^{-1}
            acceptance_prob = torch.max(acceptance_prob, epsilon)
            if torch.pow(acceptance_prob, a) < threshold:
                # stop if the acceptance probability crosses the threshold
                break
        self.step_size = step_size

    def _adapt_step_size(self, node, world, iteration_number):
        """
        Adapt the step size based on the acceptance probabity of the previous sample
        Calculated using Algorithm 6: NUTS with Dual Averaging
        http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param ratio: the ratio that we want to converge
        :param iteration_number: the current iteration of inference
        """
        closeness_frac = 1.0 / (iteration_number + self.t)
        self.closeness = ((1 - closeness_frac) * self.closeness) + (
            closeness_frac * (self.optimal_acceptance_prob - self.ratio)
        )

        log_step_size = self.mu - (
            (math.sqrt(iteration_number) / self.gamma) * self.closeness
        )

        step_frac = iteration_number ** (-self.kappa)
        log_best_step_size = (step_frac * log_step_size) + (
            (1 - step_frac) * math.log(self.best_step_size)
        )

        self.step_size = math.exp(log_step_size)
        self.best_step_size = math.exp(log_best_step_size)

    def _adapt_mass_matrix(self, node: RVIdentifier, world: World, iteration: int):
        """
        The mass matrix is approximated using the covariance of the samples
        Calculated using online covariance algorithm
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

        :param node: the node for which we have already proposed a new value for
        :param world: the world in which we have already proposed a new value
        for node
        :param iteration: the current iteration of inference
        """
        if iteration <= self.start_cov:
            return

        # detach everything
        if self.sample_mean is not None:
            self.sample_mean = self.sample_mean.detach()
        if self.track_covariance is not None:
            self.track_covariance = self.track_covariance.detach()
        if self.covariance is not None:
            self.covariance = self.covariance.detach()
        if self.l_inv is not None:
            self.l_inv = self.l_inv.detach()

        # calculate the iteration for the warmup window
        window_iteration = (
            iteration - self.start_cov - self.window_size + self.initial_window_size
        )

        # calculate co-moment
        node_var = world.variables_.get_node_raise_error(node)
        sample = node_var.transformed_value
        sample_vector = torch.reshape(sample, (-1,))
        if self.sample_mean is None:
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

        self.track_covariance = self.co_moment / window_iteration

        # update mass matrix at the end of the window
        if window_iteration == self.window_size:
            # update mass matrix and l_inv
            self.covariance = self.track_covariance.detach()
            self.covariance = (
                self.covariance
                + torch.eye(len(self.covariance)) * self.covariance_diagonal_padding
            )
            lower = torch.linalg.cholesky(self.covariance)
            identity_matrix = torch.eye(lower.size(-1), dtype=lower.dtype)
            self.l_inv = torch.triangular_solve(
                identity_matrix, lower, upper=False
            ).solution
            # reset sample mean and covariance
            self.sample_mean = torch.zeros(len(sample_vector), dtype=sample.dtype)
            self.co_moment = torch.zeros(
                len(sample_vector), len(sample_vector), dtype=sample.dtype
            )
            # update window size
            self.window_size *= 2

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
        The adaptation uses warmup phases as described by Stan
        https://mc-stan.org/docs/2_25/reference-manual/hmc-algorithm-parameters.html
        Phase 1: start_cov initial iterations for adapting the step size
        Phase 2: growing slow intervals starting with initial_window_size
            for adapting the step size and covariance
        Phase 3: remaining iterations for adapting the step size

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

        if not self.initialized:
            self._find_reasonable_step_size(node, world)
            self.mu = math.log(10 * self.step_size)
            self.best_step_size = 1.0
            self.initialized = True

        self._adapt_step_size(node, world, iteration_number)
        if self.use_dense_mass_matrix:
            self._adapt_mass_matrix(node, world, iteration_number)

        if iteration_number == num_adaptive_samples:
            self.step_size = self.best_step_size

    def _compute_kinetic_energy(self, r: Tensor) -> Tensor:
        """
        Compute the kinetic energy with the given momentum

        :param r: the momentum
        :returns: the kinetic energy calculated by 1/2 * r.T * cov * r
        """
        r_vector = torch.reshape(r, (-1,))
        if self.use_dense_mass_matrix:
            return torch.matmul(r_vector.T, torch.matmul(self.covariance, r_vector)) / 2
        return (r_vector * r_vector).sum() / 2

    def _compute_likelihood(self, node, world, theta):
        """
        Compute the likelihood theta

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param theta: the proposed transformed value for the node
        """
        score = world.propose_change_transformed_value(
            node, theta, allow_graph_update=False
        )[3]

        world.reset_diff()
        return score

    def _compute_potential_energy_gradient(self, node, world, theta):
        """
        Compute the gradient of the likelihood w.r.t the new transformed value

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param theta: the proposed transformed value for the node
        """
        score = world.propose_change_transformed_value(
            node, theta, allow_graph_update=False
        )[3]
        node_var = world.get_node_in_world(node, False)
        is_valid, grad_U = compute_first_gradient(
            -score, node_var.transformed_value, retain_graph=True
        )
        score = score.detach()
        world.reset_diff()
        return is_valid, grad_U

    def _compute_hamiltonian(self, node, world, theta, r):
        """
        Compute the Hamiltonian with the given theta and r

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param theta: the proposed transformed value for the node
        :param r: the momentum of the node
        """
        kinetic_energy = self._compute_kinetic_energy(r)
        likelihood = self._compute_likelihood(node, world, theta)
        return likelihood - kinetic_energy

    def _leapfrog_step(self, node, world, theta, r, step_size):
        """
        Compute the Hamiltonian with the given theta and r

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param theta: the proposed transformed value for the node
        :param r: the momentum of the node
        :param step_size: the leapfrog step size
        """
        is_valid, grad_U = self._compute_potential_energy_gradient(node, world, theta)
        if not is_valid:
            return is_valid, theta, r
        r = r - step_size * grad_U / 2

        if self.use_dense_mass_matrix:
            r_vector = torch.reshape(r, (-1,))
            if self.covariance is None:
                self.covariance = torch.eye(
                    len(r_vector), dtype=theta.dtype, device=theta.device
                )
                self.covariance = self.covariance.detach()
            r_scaled = torch.reshape(torch.matmul(self.covariance, r_vector), r.shape)
        else:
            r_scaled = r

        theta1 = theta + step_size * r_scaled
        is_valid, grad_U = self._compute_potential_energy_gradient(node, world, theta1)
        if not is_valid:
            return is_valid, theta1, r
        r1 = r - step_size * grad_U / 2

        return is_valid, theta1, r1

    def _build_tree_base_case(self, node, world, build_tree_input: Tuple):
        """
        Base case of BuildTree as denoted in Algorithm 6

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param build_tree_input: the input parameters of build_tree
        """
        theta, r, u, v, j, theta0, r0 = build_tree_input
        is_valid, theta1, r1 = self._leapfrog_step(
            node, world, theta, r, v * self.step_size
        )
        if not is_valid:
            return (
                theta,
                r,
                theta,
                r,
                theta,
                tensor(1.0, dtype=theta.dtype),
                tensor(0.0, dtype=theta.dtype),
                tensor(0.0, dtype=theta.dtype),
                tensor(1.0, dtype=theta.dtype),
            )

        h1 = self._compute_hamiltonian(node, world, theta1, r1)
        h0 = self._compute_hamiltonian(node, world, theta0, r0)
        dH = h1 - h0

        n1 = (u <= dH).to(dtype=theta1.dtype)
        s1 = (u < self.delta_max + dH).to(dtype=theta1.dtype)
        accept_ratio = torch.min(
            tensor(1.0, dtype=theta1.dtype, device=theta1.device), torch.exp(dH)
        )
        return (
            theta1,
            r1,
            theta1,
            r1,
            theta1,
            n1,
            s1,
            accept_ratio,
            tensor(1.0, dtype=theta1.dtype),
        )

    def _build_tree(self, node: RVIdentifier, world: World, build_tree_input: Tuple):
        """
        BuildTree as denoted in Algorithm 6

        :param node: the node for which we'll need to propose a new value for
        :param world: the world in which we'll propose a new value for node
        :param build_tree_input: the input parameters of build_tree
        """
        theta, r, u, v, j, theta0, r0 = build_tree_input
        if j == 0:
            return self._build_tree_base_case(node, world, build_tree_input)
        else:
            subtree_input = theta, r, u, v, j - 1, theta0, r0
            subtree_output = self._build_tree(node, world, subtree_input)
            theta_n, r_n, theta_p, r_p, theta1, n1, s1, a1, na1 = subtree_output

            if torch.eq(s1, tensor(1.0, dtype=s1.dtype, device=s1.device)):
                if v < 0:
                    neg_output = self._build_tree(
                        node, world, (theta_n, r_n, u, v, j - 1, theta0, r0)
                    )
                    theta_n, r_n, _, _, theta2, n2, s2, a2, na2 = neg_output
                else:
                    pos_output = self._build_tree(
                        node, world, (theta_p, r_p, u, v, j - 1, theta0, r0)
                    )
                    _, _, theta_p, r_p, theta2, n2, s2, a2, na2 = pos_output

                # sometimes n1 + n2 is 0
                change_val = dist.Bernoulli(
                    n2
                    / torch.max(
                        n1 + n2, tensor(1.0, dtype=theta.dtype, device=theta.device)
                    )
                ).sample()
                if change_val:
                    theta1 = theta2
                a1 = a1 + a2
                na1 = na1 + na2

                if torch.ne(s2, tensor(1.0, dtype=s2.dtype, device=s2.device)):
                    s1 = s2
                else:
                    if self.use_dense_mass_matrix:
                        p_vector = torch.reshape(theta_p, (-1,))
                        if self.l_inv is None:
                            self.l_inv = torch.eye(
                                len(p_vector), dtype=theta.dtype, device=theta.device
                            )
                        transformed_p = torch.reshape(
                            torch.matmul(self.l_inv, p_vector), theta_p.shape
                        )
                        n_vector = torch.reshape(theta_p, (-1,))
                        transformed_n = torch.reshape(
                            torch.matmul(self.l_inv, n_vector), theta_n.shape
                        )
                    else:
                        transformed_p = theta_p
                        transformed_n = theta_n

                    neg_turn = ((transformed_p - transformed_n) * r_n).sum() >= 0
                    pos_turn = ((transformed_p - transformed_n) * r_p).sum() >= 0
                    s1 = neg_turn * pos_turn
                n1 = n1 + n2
            return theta_n, r_n, theta_p, r_p, theta1, n1, s1, a1, na1

    def _initialize_momentum(self, theta: Tensor) -> Tensor:
        """
        Initialize momentum

        :param theta: the position vector
        """
        if self.use_dense_mass_matrix:
            # initialize momentum
            if self.covariance is None:
                self.covariance = torch.eye(
                    len(theta.reshape(-1)), dtype=theta.dtype, device=theta.device
                )
            r = (
                dist.MultivariateNormal(
                    torch.zeros(
                        len(theta.reshape(-1)), dtype=theta.dtype, device=theta.device
                    ),
                    precision_matrix=self.covariance,
                )
                .sample()
                .reshape(theta.shape)
            )
        else:
            r = torch.randn(theta.shape, dtype=theta.dtype, device=theta.device)
        return r

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
        theta = node_var.transformed_value

        if not self.initialized:
            self._find_reasonable_step_size(node, world)
            self.mu = math.log(10 * self.step_size)
            self.best_step_size = 1.0
            self.initialized = True
        r = self._initialize_momentum(theta)

        u = (
            dist.Uniform(tensor(0.0, dtype=theta.dtype, device=theta.device), 1.0)
            .sample()
            .log()
        )

        theta_n = theta
        theta_p = theta
        r_n = r
        r_p = r
        theta_propose = theta
        n = tensor(1.0, dtype=theta.dtype)
        s = tensor(1.0, dtype=theta.dtype)

        for j in range(self.max_depth):
            v = (
                dist.Bernoulli(
                    tensor(0.5, dtype=theta.dtype, device=theta.device)
                ).sample()
            ) * 2 - 1
            if v < 0:
                build_tree_output = self._build_tree(
                    node, world, (theta_n, r_n, u, v, j, theta, r)
                )
                theta_n, r_n, _, _, theta1, n1, s1, a, na = build_tree_output
            else:
                build_tree_output = self._build_tree(
                    node, world, (theta_p, r_p, u, v, j, theta, r)
                )
                _, _, theta_p, r_p, theta1, n1, s1, a, na = build_tree_output

            if torch.eq(s1, tensor(1.0, dtype=theta.dtype)):
                change_val = dist.Bernoulli(
                    torch.min(
                        tensor(1.0, dtype=theta.dtype, device=theta.device), n1 / n
                    )
                ).sample()
                if change_val:
                    theta_propose = theta1
            n = n + n1

            if torch.ne(s1, tensor(1.0, dtype=s1.dtype, device=s1.device)):
                s = s1
            else:
                if self.use_dense_mass_matrix:
                    p_vector = torch.reshape(theta_p, (-1,))
                    if self.l_inv is None:
                        self.l_inv = torch.eye(
                            len(p_vector), dtype=theta.dtype, device=theta.device
                        )
                    transformed_p = torch.reshape(
                        torch.matmul(self.l_inv, p_vector), theta_p.shape
                    )
                    n_vector = torch.reshape(theta_p, (-1,))
                    transformed_n = torch.reshape(
                        torch.matmul(self.l_inv, n_vector), theta_n.shape
                    )
                else:
                    transformed_p = theta_p
                    transformed_n = theta_n

                turn_n = ((transformed_p - transformed_n) * r_n).sum() >= 0
                turn_p = ((transformed_p - transformed_n) * r_p).sum() >= 0
                s = turn_n * turn_p

            if torch.ne(s, tensor(1.0, dtype=s.dtype, device=s.device)):
                break

        # need this for adaptive step (sometimes na is 0)
        self.ratio = a.item() / max(na.item(), 1.0)

        q = node_var.inverse_transform_value(theta_propose)
        (children_log_update, _, node_log_update, _) = world.propose_change(
            node, q, allow_graph_update=False
        )
        world.reset_diff()

        # cancel out children and node log update because not needed for NUTS
        return q.detach(), -children_log_update - node_log_update, {}

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
