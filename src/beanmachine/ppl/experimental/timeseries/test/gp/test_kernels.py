# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from numbers import Number

import gpytorch
import pytest
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel
from gpytorch.priors import GammaPrior, LogNormalPrior
from sts.gp.kernels import ChangePointKernel, ConstantKernel

torch.manual_seed(0)
SAMPLE_DATA = torch.linspace(-3, 3, 100)[:, None]


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_constant_kernel(x):
    k = ConstantKernel(
        constant=1.0,
        constant_prior=LogNormalPrior(0, 1),
        constant_constraint=GreaterThan(-1),
    )
    assert torch.allclose(k.constant, torch.tensor(1.0))
    assert type(k.constant_prior) == LogNormalPrior
    assert type(k.raw_constant_constraint) == GreaterThan


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_changepoint_kernel(x):
    base_k1 = gpytorch.kernels.MaternKernel(lengthscales=0.2)
    base_k2 = ConstantKernel(constant=0.0)
    location = 1.0
    steep = 2.0
    k = ChangePointKernel(
        (base_k1, base_k2),
        location=location,
        steep=steep,
        location_prior=LogNormalPrior(1, 2),
        steep_prior=GammaPrior(1, 2),
    )
    assert torch.allclose(k.location, torch.tensor(location))
    assert torch.allclose(k.steep, torch.tensor(steep))
    assert type(k.location_prior) == LogNormalPrior
    assert type(k.steep_prior) == GammaPrior


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_changepoint_boundary(x):
    x1 = torch.tensor([0.1, 0.2])
    x2 = torch.tensor([0.8, 0.9])

    base_k1 = ConstantKernel(constant=100.0)
    base_k2 = ConstantKernel(constant=0.0)
    location = 0.5
    steep = 100.0
    k = ChangePointKernel(
        (base_k1, base_k2),
        location=location,
        steep=steep,
    )
    assert torch.allclose(k.location, torch.tensor(location))
    assert torch.allclose(k.steep, torch.tensor(steep))

    assert torch.allclose(k(x1, x1).evaluate(), base_k1(x1, x1).evaluate())
    assert torch.allclose(k(x2, x2).evaluate(), base_k2(x2, x2).evaluate())
    assert torch.allclose(k(x1, x2).evaluate(), torch.zeros(2, 2))


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_changewindow_kernel(x):
    base_k1 = gpytorch.kernels.MaternKernel(lengthscales=0.2)
    base_k2 = ConstantKernel(constant=0.0)
    location = (-1.0, 1.0)
    steep = (0.5, 50)
    k = ChangePointKernel(
        (base_k1, base_k2, base_k1),
        location=location,
        steep=steep,
        location_prior=LogNormalPrior(1, 2),
        steep_prior=GammaPrior(1, 2),
    )
    assert torch.allclose(k.location, torch.tensor(location))
    assert torch.allclose(k.steep, torch.tensor(steep))
    assert type(k.location_prior) == LogNormalPrior
    assert type(k.steep_prior) == GammaPrior


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_changewindow_boundary(x):
    x1 = torch.tensor([0.1, 0.2])
    x2 = torch.tensor([0.8, 0.9])
    x3 = torch.tensor([1.1, 1.2])

    base_k1 = ConstantKernel(constant=100.0)
    base_k2 = ConstantKernel(constant=0.0)
    location = (0.5, 1.0)
    steep = 100.0
    k = ChangePointKernel(
        (base_k1, base_k2, base_k1),
        location=location,
        steep=steep,
    )

    assert torch.allclose(k.location, torch.tensor(location))
    assert torch.allclose(k.steep, torch.tensor(steep))

    assert torch.allclose(k(x1, x1).evaluate(), base_k1(x1, x1).evaluate())
    assert torch.allclose(k(x2, x2).evaluate(), base_k2(x2, x2).evaluate())
    assert torch.allclose(k(x3, x3).evaluate(), base_k1(x3, x3).evaluate())
    assert torch.allclose(k(x1, x2).evaluate(), torch.zeros(2, 2))
    assert torch.allclose(k(x2, x3).evaluate(), torch.zeros(2, 2))


@pytest.mark.parametrize("changepoint", [0.0, 1.0, [0.0, 1.0]])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
@pytest.mark.parametrize("last_dim_is_batch", [False, True])
@pytest.mark.parametrize("batched_input", [False, True])
def test_changepoint_batched_shape(
    changepoint, batch_shape, last_dim_is_batch, batched_input
):
    num_changepoints = 1 if isinstance(changepoint, Number) else len(changepoint)
    kernels = [RBFKernel() for _ in range(num_changepoints + 1)]
    cp = ChangePointKernel(kernels, location=changepoint, batch_shape=batch_shape)
    assert cp.location.shape == (*batch_shape, 1, 1, num_changepoints)
    N = 10
    K = 5  # batched last dim
    input_shape_pref = batch_shape if batched_input else ()
    input_shape = input_shape_pref + (N, 1)
    if last_dim_is_batch:
        input_shape = input_shape[:-1] + (K,)
    x1 = torch.randn(input_shape)
    x2 = torch.randn(input_shape)
    k_12 = cp(x1, x2, last_dim_is_batch=last_dim_is_batch).evaluate()
    expected_shape = (
        (*batch_shape, N, N) if not last_dim_is_batch else (*batch_shape, K, N, N)
    )
    assert k_12.shape == expected_shape
