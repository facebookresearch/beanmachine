import math

import torch


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
