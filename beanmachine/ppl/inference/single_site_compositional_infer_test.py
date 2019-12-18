# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.inference.single_site_compositional_infer import (
    SingleSiteCompositionalInference,
)
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


class SingleSiteCompositionalInferenceTest(unittest.TestCase):
    class SampleModel(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def foobar(self):
            return dist.Categorical(tensor([0.5, 0, 5]))

        @sample
        def foobaz(self):
            return dist.Bernoulli(0.1)

        @sample
        def bazbar(self):
            return dist.Poisson(tensor([4]))

    def test_single_site_compositionl_inference(self):
        model = self.SampleModel()
        c = SingleSiteCompositionalInference()
        foo_key = model.foo()
        c.world_ = World()
        distribution = dist.Bernoulli(0.1)
        val = distribution.sample()
        c.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteUniformProposer
            ),
            True,
        )

        distribution = dist.Normal(tensor(0.0), tensor(1.0))
        val = distribution.sample()
        c.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )

        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key),
                SingleSiteNewtonianMonteCarloProposer,
            ),
            True,
        )

        distribution = dist.Categorical(tensor([0.5, 0, 5]))
        val = distribution.sample()
        c.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )

        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteUniformProposer
            ),
            True,
        )

        distribution = dist.Poisson(tensor([4.0]))
        val = distribution.sample()
        c.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteAncestralProposer
            ),
            True,
        )

    def test_single_site_compositionl_inference_with_input(self):
        model = self.SampleModel()
        c = SingleSiteCompositionalInference({model.foo: SingleSiteAncestralProposer})
        foo_key = model.foo()
        c.world_ = World()
        distribution = dist.Normal(0.1, 1)
        val = distribution.sample()
        c.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteAncestralProposer
            ),
            True,
        )
