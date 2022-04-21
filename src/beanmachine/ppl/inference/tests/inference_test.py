# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World, init_from_prior


class SampleModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(0.0, 1.0)

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), 1.0)

    @bm.functional
    def baz(self):
        return self.bar() * 2.0


class SampleDoubleModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(torch.tensor(0.0).double(), torch.tensor(1.0).double())

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), torch.tensor(1.0).double())


@pytest.mark.parametrize("multiprocess", [False, True])
def test_inference(multiprocess):
    if multiprocess and sys.platform.startswith("win"):
        pytest.skip(
            "Windows does not support fork-based multiprocessing (which is necessary "
            "for running parallel inference within pytest."
        )

    model = SampleModel()
    mh = bm.SingleSiteAncestralMetropolisHastings()
    queries = [model.foo(), model.baz()]
    observations = {model.bar(): torch.tensor(0.5)}
    num_samples = 30
    num_chains = 2
    samples = mh.infer(
        queries,
        observations,
        num_samples,
        num_adaptive_samples=num_samples,
        num_chains=num_chains,
        run_in_parallel=multiprocess,
        mp_context="fork",
    )

    assert model.foo() in samples
    assert isinstance(samples[model.foo()], torch.Tensor)
    assert samples[model.foo()].shape == (num_chains, num_samples)
    assert samples.get_num_samples(include_adapt_steps=True) == num_samples * 2
    # make sure that the RNG state for each chain is different
    assert not torch.equal(
        samples.get_chain(0)[model.foo()], samples.get_chain(1)[model.foo()]
    )


def test_get_proposers():
    world = World()
    model = SampleModel()
    world.call(model.bar())
    nuts = bm.GlobalNoUTurnSampler()
    proposers = nuts.get_proposers(world, world.latent_nodes, 10)
    assert all(isinstance(proposer, BaseProposer) for proposer in proposers)


def test_initialize_world():
    model = SampleModel()
    world = World()._initialize_world([model.bar()])
    assert model.foo() in world
    assert model.bar() in world


def test_initialize_from_prior():
    model = SampleModel()
    queries = [model.foo()]

    samples_from_prior = []
    for _ in range(10000):
        world = World(initialize_fn=init_from_prior)._initialize_world(queries)
        val = world.get(model.foo())
        samples_from_prior.append(val.item())

    assert samples_from_prior[0] != samples_from_prior[1]
    assert math.isclose(sum(samples_from_prior) / 10000.0, 0.0, abs_tol=1e-2)


def test_initialization_resampling():
    mh = bm.SingleSiteAncestralMetropolisHastings()

    @bm.random_variable
    def foo():
        return dist.Uniform(3.0, 5.0)

    # verify that the method re-sample as expected
    retries = 0

    def init_after_three_tries(world: World, rv: RVIdentifier):
        d, _ = world._run_node(rv)
        nonlocal retries
        retries += 1
        return torch.tensor(float("nan")) if retries < 3 else d.sample()

    sampler = mh.sampler(
        [foo()], {}, num_samples=10, initialize_fn=init_after_three_tries
    )
    for world in sampler:
        assert not torch.isinf(world.log_prob()) and not torch.isnan(world.log_prob())

    # an extreme case where the init value is always out of the support
    def init_to_zero(world: World, rv: RVIdentifier):
        d, _ = world._run_node(rv)
        return torch.zeros_like(d.sample())

    with pytest.raises(ValueError, match="Cannot find a valid initialization"):
        mh.infer([foo()], {}, num_samples=10, initialize_fn=init_to_zero)


@pytest.mark.parametrize(
    "algorithm",
    [
        bm.GlobalNoUTurnSampler(),
        bm.GlobalHamiltonianMonteCarlo(trajectory_length=1.0),
        bm.SingleSiteAncestralMetropolisHastings(),
        bm.SingleSiteNewtonianMonteCarlo(),
        bm.SingleSiteUniformMetropolisHastings(),
    ],
)
def test_inference_with_double_dtype(algorithm):
    model = SampleDoubleModel()
    queries = [model.foo()]
    bar_val = torch.tensor(0.5).double()
    # make sure that the inference can run successfully
    samples = algorithm.infer(
        queries,
        {model.bar(): bar_val},
        num_samples=20,
        num_chains=1,
    )
    assert samples[model.foo()].dtype == bar_val.dtype
