# Copyright (c) Facebook, Inc. and its affiliates
import unittest
from typing import List, Optional

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.neutra.iafmcmc_proposer import IAFMCMCProposer
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import Variable, World
from beanmachine.ppl.world.world import TransformType
from torch import nn, tensor as tensor


class IAFJacobainTest(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.zeros(2), torch.ones(2))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.ones(2))

    class CustomInference(AbstractMHInference):
        def __init__(
            self,
            proposer,
            transform_type: TransformType = TransformType.DEFAULT,
            transforms: Optional[List] = None,
        ):
            super().__init__(
                proposer, transform_type=transform_type, transforms=transforms
            )
            self.proposer_ = proposer

        def find_best_single_site_proposer(self, node: RVIdentifier):
            return self.proposer_

    def setUp(self):
        torch.manual_seed(11)

    def optimizer_func(self, lr, weight_decay):
        return lambda parameters: torch.optim.Adam(
            parameters, lr=1e-4, weight_decay=1e-5
        )

    def test_jacobain_for_iaf_proposer(self):
        world = World()
        model = self.SampleModel()
        foo_key = model.foo()
        bar_key = model.bar()

        world.set_observations({bar_key: tensor([0.1, 0.1])})
        world_vars = world.variables_.vars()

        world_vars[foo_key] = Variable(
            distribution=dist.Normal(torch.zeros(2), torch.ones(2)),
            value=tensor([0.5, 0.5]),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(
                tensor([0.5, 0.5])[0]
            )
            + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor([0.5, 0.5])[1]),
            children=set({bar_key}),
            transformed_value=tensor([0.5, 0.5]),
            jacobian=tensor(0.0),
        )

        world_vars[bar_key] = Variable(
            distribution=dist.Normal(tensor([0.5, 0.5]), torch.ones(2)),
            value=tensor([0.1, 0.1]),
            log_prob=dist.Normal(tensor([0.5, 0.5])[0], tensor(0.1)).log_prob(
                tensor([0.1, 0.1])[0]
            )
            + dist.Normal(tensor([0.5, 0.5])[1], tensor(0.1)).log_prob(
                tensor([0.1, 0.1])[1]
            ),
            parent=set({foo_key}),
            transformed_value=tensor([0.1, 0.1]),
            jacobian=tensor(0.0),
        )
        # all parameters needed in IAF proposer
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        length = 2
        # pass the parameters in to construct IAF
        optimizer_func = self.optimizer_func(1e-4, 1e-5)

        training_sample_size = 100

        iaf_proposer = IAFMCMCProposer(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            [],
        )
        for i in range(1, 1000):
            iaf_proposer.do_adaptation(foo_key, world, tensor(1.0), 1000, True, i)
        infer = self.CustomInference(iaf_proposer)
        infer.queries_ = [model.foo()]
        infer.initialize_world(initialize_from_prior=True)

        is_accepted, acceptance_probability = infer.single_inference_run(
            foo_key, iaf_proposer
        )
        self.assertEqual(is_accepted, True)
        self.assertAlmostEqual(acceptance_probability, 1.0, delta=1e-5)
