# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from unittest.mock import patch

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.nuts_proposer import (
    NUTSProposer,
)
from beanmachine.ppl.inference.proposer.sequential_proposer import (
    SequentialProposer,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)


class SampleModel:
    @bm.random_variable
    def foo(self, i: int):
        return dist.Beta(2.0, 2.0)

    @bm.random_variable
    def bar(self, i: int):
        return dist.Bernoulli(self.foo(i))

    @bm.random_variable
    def baz(self):
        return dist.Normal(self.bar(0) + self.bar(1), 1.0)


class ChangingSupportSameShapeModel:
    # the support of `component` is changing, but (because we indexed alpha
    # by k) all random_variables have the same shape
    @bm.random_variable
    def K(self):
        return dist.Poisson(rate=2.0)

    @bm.random_variable
    def alpha(self, k):
        return dist.Dirichlet(torch.ones(k))

    @bm.random_variable
    def component(self, i):
        alpha = self.alpha(self.K().int().item() + 1)
        return dist.Categorical(alpha)


class ChangingShapeModel:
    # here since we did not index alpha, its shape in each world is changing
    @bm.random_variable
    def K(self):
        return dist.Poisson(rate=2.0)

    @bm.random_variable
    def alpha(self):
        return dist.Dirichlet(torch.ones(self.K().int().item() + 1))

    @bm.random_variable
    def component(self, i):
        return dist.Categorical(self.alpha())


def test_inference_config():
    model = SampleModel()
    nuts = bm.GlobalNoUTurnSampler()
    compositional = bm.CompositionalInference({model.foo: nuts})
    queries = [model.foo(0), model.foo(1)]
    observations = {model.baz(): torch.tensor(2.0)}

    # verify that inference can run without error
    compositional.infer(queries, observations, num_chains=1, num_samples=10)

    # verify that proposers are spawned correctly
    world = compositional._initialize_world(queries, observations)
    with patch.object(nuts, "get_proposers", wraps=nuts.get_proposers) as mock:
        proposers = compositional.get_proposers(
            world, target_rvs=world.latent_nodes, num_adaptive_sample=0
        )
        # NUTS should receive {foo(0), foo(1)} as its target rvs
        mock.assert_called_once_with(world, {model.foo(0), model.foo(1)}, 0)
    # there should be one NUTS proposer for both foo(0) and foo(1), one ancestral MH
    # proposer for bar(0), and another ancestral MH proposer for bar(1)
    assert len(proposers) == 3
    # TODO: find a way to validate the proposer instead of relying on the order of
    # return value
    assert isinstance(proposers[0], NUTSProposer)
    assert proposers[0]._target_rvs == {model.foo(0), model.foo(1)}
    # the rest of nodes are updated by default proposers (uniform proposer for bernoulli)
    assert isinstance(proposers[1], SingleSiteUniformProposer)
    assert isinstance(proposers[2], SingleSiteUniformProposer)
    assert {proposers[1].node, proposers[2].node} == {model.bar(0), model.bar(1)}

    # test overriding default kwarg
    compositional = bm.CompositionalInference(
        {
            model.foo: bm.GlobalNoUTurnSampler(),
            ...: bm.SingleSiteAncestralMetropolisHastings(),
        }
    )
    compositional.infer(queries, observations, num_chains=1, num_samples=2)
    world = compositional._initialize_world(queries, observations)
    proposers = compositional.get_proposers(
        world, target_rvs=world.latent_nodes, num_adaptive_sample=0
    )
    assert isinstance(proposers[0], NUTSProposer)
    assert isinstance(proposers[1], SingleSiteAncestralProposer)
    assert isinstance(proposers[2], SingleSiteAncestralProposer)
    assert {proposers[1].node, proposers[2].node} == {model.bar(0), model.bar(1)}


def test_config_inference_with_tuple_of_rv():
    model = SampleModel()
    nuts = bm.GlobalNoUTurnSampler()
    compositional = bm.CompositionalInference({(model.foo, model.baz): nuts})
    world = compositional._initialize_world([model.baz()], {})
    with patch.object(nuts, "get_proposers", wraps=nuts.get_proposers) as mock:
        compositional.get_proposers(
            world, target_rvs=world.latent_nodes, num_adaptive_sample=10
        )
        # NUTS should receive {foo(0), foo(1), model.baz()} as its target rvs
        mock.assert_called_once_with(
            world, {model.foo(0), model.foo(1), model.baz()}, 10
        )


def test_config_inference_with_tuple_of_inference():
    model = SampleModel()
    compositional = bm.CompositionalInference(
        {
            (model.foo, model.bar): (
                bm.SingleSiteAncestralMetropolisHastings(),
                bm.SingleSiteUniformMetropolisHastings(),
            ),
            model.baz: bm.GlobalNoUTurnSampler(),
        }
    )
    # verify that inference can run without error
    compositional.infer([model.baz()], {}, num_chains=1, num_samples=10)
    # examine the proposer types
    world = compositional._initialize_world([model.baz()], {})
    proposers = compositional.get_proposers(
        world, target_rvs=world.latent_nodes, num_adaptive_sample=10
    )
    assert len(proposers) == 2
    sequential_proposer = proposers[int(isinstance(proposers[0], NUTSProposer))]
    assert isinstance(sequential_proposer, SequentialProposer)
    assert len(sequential_proposer.proposers) == 4
    proposer_count = Counter(map(type, sequential_proposer.proposers))
    assert proposer_count[SingleSiteAncestralProposer] == 2


def test_nested_compositional_inference():
    model = SampleModel()
    ancestral_mh = bm.SingleSiteAncestralMetropolisHastings()
    compositional = bm.CompositionalInference(
        {
            (model.foo, model.bar): bm.CompositionalInference(
                {
                    model.foo: bm.GlobalNoUTurnSampler(),
                    # this ancestral mh class is never going to be invoked
                    model.baz: ancestral_mh,
                }
            )
        }
    )

    with patch.object(
        ancestral_mh, "get_proposers", wraps=ancestral_mh.get_proposers
    ) as mock:
        # verify that inference can run without error
        compositional.infer([model.baz()], {}, num_chains=1, num_samples=10)

        # the ancestral_mh instance shouldn't been invoked at all
        mock.assert_not_called()


def test_block_inference_with_default_algorithm():
    model = SampleModel()
    # block foo and baz together, but uses the default inference
    compositional = bm.CompositionalInference({(model.foo, model.bar, model.baz): ...})
    # make sure that things can run without failure
    queries = [model.baz()]
    observations = {}
    compositional.infer(queries, observations, num_chains=1, num_samples=10)
    # check to see if proposers are indeed blocked together
    world = compositional._initialize_world(queries, observations)
    proposers = compositional.get_proposers(world, world.latent_nodes, 0)
    assert len(proposers) == 1
    assert isinstance(proposers[0], SequentialProposer)


def test_block_inference_changing_support():
    model = ChangingSupportSameShapeModel()
    queries = [model.K()] + [model.component(j) for j in range(3)]
    compositional = bm.CompositionalInference(
        {(model.K, model.component): bm.SingleSiteAncestralMetropolisHastings()}
    )
    sampler = compositional.sampler(queries, {}, num_samples=10, num_adaptive_samples=5)
    old_world = next(sampler)
    for world in sampler:  # this should run without failing
        # since it's actually possible to sample two identical values, we need
        # to check for tensor identity
        if world[model.K()] is not old_world[model.K()]:
            # if one of the node in a block is updated, the rest of the nodes should
            # also been updated
            for i in range(3):
                assert world[model.component(i)] is not old_world[model.component(i)]
        else:
            # just as a sanity check to show that the tensor identity check is doing
            # what we expected
            assert world[model.component(0)] is old_world[model.component(0)]
        old_world = world


def test_block_inference_changing_shape():
    model = ChangingShapeModel()
    queries = [model.K()] + [model.component(j) for j in range(3)]
    compositional = bm.CompositionalInference()

    # should run without error
    samples = compositional.infer(queries, {}, num_samples=5, num_chains=1).get_chain()
    # however, K is going to be stuck at its initial value because changing it will
    # invalidate alpha
    assert torch.all(samples[model.K()] == samples[model.K()][0])
