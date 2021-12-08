import warnings

import beanmachine.ppl as bm
import numpy as np
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.hmc_utils import (
    DualAverageAdapter,
    MassMatrixAdapter,
    WelfordCovariance,
    WindowScheme,
    RealSpaceTransform,
)
from beanmachine.ppl.world import World


class SampleModel:
    @bm.random_variable
    def foo(self):
        return dist.Uniform(0.0, 1.0)

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), 1.0)


def test_dual_average_adapter():
    adapter = DualAverageAdapter(0.1)
    epsilon1 = adapter.step(torch.tensor(1.0))
    epsilon2 = adapter.step(torch.tensor(0.0))
    assert epsilon2 < adapter.finalize() < epsilon1


def test_dual_average_with_different_delta():
    adapter1 = DualAverageAdapter(1.0, delta=0.8)
    adapter2 = DualAverageAdapter(1.0, delta=0.2)
    prob = torch.tensor(0.5)
    # prob > delta means we can increase the step size, wherease prob < delta means
    # we need to decrease the step size
    epsilon1 = adapter1.step(prob)
    epsilon2 = adapter2.step(prob)
    assert epsilon1 < epsilon2


def test_small_window_scheme():
    num_adaptive_samples = 10
    scheme = WindowScheme(num_adaptive_samples)
    for _ in range(num_adaptive_samples):
        # no window should be created if num_adaptive_samples is too small
        assert not scheme.is_in_window
        scheme.step()


def test_middle_window_scheme():
    num_adaptive_samples = 125
    scheme = WindowScheme(num_adaptive_samples)
    num_windows = 0
    for i in range(num_adaptive_samples):
        if scheme.is_in_window:
            # there should be a margin at the beginning and the end of a window
            assert i > 0
            if scheme.is_end_window:
                num_windows += 1
                assert i < num_adaptive_samples
        scheme.step()
    # there should only be a single window
    assert num_windows == 1


@pytest.mark.parametrize("num_adaptive_samples", [175, 300, 399, 543])
def test_large_window_scheme(num_adaptive_samples):
    scheme = WindowScheme(num_adaptive_samples)
    window_sizes = []
    for _ in range(num_adaptive_samples):
        if scheme.is_end_window:
            window_sizes.append(scheme._window_size)
        scheme.step()
    # size of windows should be monotonically increasing
    sorted_window_sizes = sorted(window_sizes)
    assert window_sizes == sorted_window_sizes
    for win1, win2 in zip(window_sizes[:-1], window_sizes[1:-1]):
        # except for last window, window size should keep doubling
        assert win2 == win1 * 2


def test_mass_matrix_adapter():
    model = SampleModel()
    world = World()
    world.call(model.bar())
    positions = RealSpaceTransform(world, world.latent_nodes)(dict(world))
    mass_matrix_adapter = MassMatrixAdapter()
    momentums = mass_matrix_adapter.initialize_momentums(positions)
    for node, z in positions.items():
        assert node in momentums
        assert isinstance(momentums[node], torch.Tensor)
        assert len(momentums[node]) == z.numel()
        # after seeing the nodes for the first time, the adapter should've initialized
        # the mass matrix and the distribution to generate momentum
        assert node in mass_matrix_adapter.mass_inv
        assert node in mass_matrix_adapter.momentum_dists
    mass_inv_old = mass_matrix_adapter.mass_inv.copy()
    mass_matrix_adapter.step(positions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mass_matrix_adapter.finalize()

    # mass matrix adapter has seen less than 2 samples, so mass_inv is not updated
    for node in mass_inv_old:
        assert torch.allclose(mass_inv_old[node], mass_matrix_adapter.mass_inv[node])


def test_diagonal_welford_covariance():
    samples = dist.MultivariateNormal(
        loc=torch.rand(5), scale_tril=torch.randn(5, 5).tril()
    ).sample((1000,))
    welford = WelfordCovariance(diagonal=True)
    for sample in samples:
        welford.step(sample)
    sample_var = torch.var(samples, dim=0)
    estimated_var = welford.finalize(regularize=False)
    assert torch.allclose(estimated_var, sample_var)
    regularized_var = welford.finalize(regularize=True)
    assert (torch.argsort(regularized_var) == torch.argsort(estimated_var)).all()


def test_dense_welford_covariance():
    samples = dist.MultivariateNormal(
        loc=torch.rand(5), scale_tril=torch.randn(5, 5).tril()
    ).sample((1000,))
    welford = WelfordCovariance(diagonal=False)
    for sample in samples:
        welford.step(sample)
    sample_cov = torch.from_numpy(np.cov(samples.T.numpy())).to(samples.dtype)
    estimated_cov = welford.finalize(regularize=False)
    assert torch.allclose(estimated_cov, sample_cov)
    regularized_cov = welford.finalize(regularize=True)
    assert (torch.argsort(regularized_cov) == torch.argsort(estimated_cov)).all()


def test_welford_exception():
    welford = WelfordCovariance()
    welford.step(torch.rand(5))
    with pytest.raises(RuntimeError):  # number of samples is too small
        welford.finalize()
