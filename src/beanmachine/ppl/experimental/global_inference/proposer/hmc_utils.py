import math
import warnings
from typing import Union, cast

import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.simple_world import (
    RVDict,
    SimpleWorld,
)


class WindowScheme:
    """
    Spliting adaptation iterations into a series of monotonically increasing windows,
    which can be used to learn the mass matrices in HMC.
    Reference:
    [1] "HMC algorithm parameters" from Stan Reference Manual
    https://mc-stan.org/docs/2_26/reference-manual/hmc-algorithm-parameters.html#automatic-parameter-tuning
    """

    def __init__(self, num_adaptive_samples: int):
        # from Stan
        if num_adaptive_samples < 20:
            # do not create any window for adapting mass matrix
            self._start_iter = self._end_iter = num_adaptive_samples
            self._window_size = 0
        elif num_adaptive_samples < 150:
            self._start_iter = int(0.15 * num_adaptive_samples)
            self._end_iter = int(0.9 * num_adaptive_samples)
            self._window_size = self._end_iter - self._start_iter
        else:
            self._start_iter = 75
            self._end_iter = num_adaptive_samples - 50
            self._window_size = 25

        self._iteration = 0

    @property
    def is_in_window(self):
        return self._iteration >= self._start_iter and self._iteration < self._end_iter

    @property
    def is_end_window(self):
        return self._iteration - self._start_iter == self._window_size - 1

    def step(self):
        if self.is_end_window:
            # prepare for next window
            self._start_iter = self._iteration + 1
            if self._end_iter - self._start_iter < self._window_size * 4:
                # window sizes should increase monotonically
                self._window_size = self._end_iter - self._start_iter
            else:
                self._window_size *= 2
        self._iteration += 1


class DualAverageAdapter:
    """
    The dual averaging mechanism that's introduced in [1] and was applied to HMC and
    NUTS for adapting step size in [2]. The implementation and notations follows [2].

    Reference:
    [1] Yurii Nesterov. "Primal-dual subgradient methods for convex problems" (2009).
        https://doi.org/10.1007/s10107-007-0149-x
    [2] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
        https://arxiv.org/abs/1111.4246
    """

    def __init__(self, initial_epsilon: float):
        self._log_avg_epsilon = 0.0
        self._H = 0.0
        self._mu = math.log(10 * initial_epsilon)
        self._t0 = 10
        self._delta = 0.8  # target mean accept prob
        self._gamma = 0.05
        self._kappa = 0.75
        self._m = 1.0  # iteration count

    def step(self, alpha: torch.Tensor) -> float:
        H_frac = 1.0 / (self._m + self._t0)
        self._H = ((1 - H_frac) * self._H) + H_frac * (self._delta - alpha.item())

        log_epsilon = self._mu - (math.sqrt(self._m) / self._gamma) * self._H
        step_frac = self._m ** (-self._kappa)
        self._log_avg_epsilon = (
            step_frac * log_epsilon + (1 - step_frac) * self._log_avg_epsilon
        )
        self._m += 1
        return math.exp(log_epsilon)

    def finalize(self) -> float:
        return math.exp(self._log_avg_epsilon)


class MassMatrixAdapter:
    """
    Reference:
    [1] "HMC algorithm parameters" from Stan Reference Manual
    https://mc-stan.org/docs/2_26/reference-manual/hmc-algorithm-parameters.html#euclidean-metric
    """

    def __init__(self):
        self.mass_inv: RVDict = {}  # inverse mass matrices, aka the inverse "metric"
        self.momentum_dists = {}  # distribution objects for generating momentums

        self._adapters = {}

    def initialize_momentums(self, world: SimpleWorld) -> RVDict:
        """Randomly draw momentum from MultivariateNormal(0, M). This momentum variable
        is denoted as p in [1] and r in [2]. Additionally, for nodes that are seen
        for the first time, this also initialize their (inverse) mass matrices to
        identity."""
        momentums = {}
        for node in world.latent_nodes:
            # initialize M^{-1} for nodes that are seen for the first time
            node_val = world.get_transformed(node).flatten()
            if node not in self.mass_inv:
                self.mass_inv[node] = torch.ones_like(node_val)
                self.momentum_dists[node] = dist.Normal(
                    torch.zeros_like(node_val), torch.ones_like(node_val)
                )
            momentums[node] = self.momentum_dists[node].sample()
        return momentums

    def step(self, world: SimpleWorld):
        for node in world.latent_nodes:
            if node not in self._adapters:
                self._adapters[node] = WelfordCovariance(diagonal=True)
            self._adapters[node].step(world.get_transformed(node).flatten())

    def finalize(self) -> None:
        for node, adapter in self._adapters.items():
            try:
                mass_inv = adapter.finalize()
                self.momentum_dists[node] = dist.Normal(
                    torch.zeros_like(mass_inv), torch.sqrt(mass_inv).reciprocal()
                )
                self.mass_inv[node] = mass_inv
            except RuntimeError as e:
                warnings.warn(str(e))
                continue
        # reset adapters to get ready for the next window
        self._adapters = {}


class WelfordCovariance:
    """
    An implementation of Welford's online algorithm for estimating the (co)variance of
    samples.
    Reference:
    [1] "Algorithms for calculating variance" on Wikipedia
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, diagonal: bool = True):
        self._mean: Union[float, torch.Tensor] = 0.0
        self._count = 0
        self._M2: Union[float, torch.Tensor] = 0.0
        self._diagonal = diagonal

    def step(self, sample: torch.Tensor) -> None:
        self._count += 1
        delta = sample - self._mean
        self._mean += delta / self._count
        delta2 = sample - self._mean
        if self._diagonal:
            self._M2 += delta * delta2
        else:
            self._M2 += torch.outer(delta, delta2)

    def finalize(self, regularize: bool = True) -> torch.Tensor:
        if self._count < 2:
            raise RuntimeError(
                "Number of samples is too small to estimate the (co)variance"
            )
        covariance = cast(torch.Tensor, self._M2) / (self._count - 1)
        if not regularize:
            return covariance

        # from Stan: regularize mass matrix for numerical stability
        covariance *= self._count / (self._count + 5.0)
        padding = 1e-3 * 5.0 / (self._count + 5.0)
        # bring covariance closer to a unit diagonal mass matrix
        if self._diagonal:
            covariance += padding
        else:
            covariance += padding * torch.eye(covariance.shape[0])

        return covariance
