# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from torch import tensor


class SingleSiteHamiltonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class ChangingParentsModel(object):
        @bm.random_variable
        def parent_one(self):
            return dist.Normal(0.0, torch.tensor(1.0))

        @bm.random_variable
        def parent_two(self):
            return dist.Normal(0.0, torch.tensor(1.0))

        @bm.random_variable
        def bad_child(self):
            if self.parent_one() > 0:
                return dist.Normal(self.parent_one(), torch.tensor(1.0))
            else:
                return dist.Normal(self.parent_two(), torch.tensor(1.0))

    def test_single_site_hamiltonian_monte_carlo(self):
        model = self.SampleNormalModel()
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1, 10)
        foo_key = model.foo()
        bar_key = model.bar()
        hmc.queries_ = [model.foo()]
        hmc.observations_ = {model.bar(): torch.tensor(0.0)}
        hmc._infer(10)

        world_vars = hmc.world_.variables_.vars()
        self.assertEqual(foo_key in world_vars, True)
        self.assertEqual(bar_key in world_vars, True)
        self.assertEqual(foo_key in world_vars[bar_key].parent, True)
        self.assertEqual(bar_key in world_vars[foo_key].children, True)

    def test_single_site_hamiltonian_monte_carlo_parents_changed(self):
        model = self.ChangingParentsModel()
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1, 10)

        parent_one_key = model.parent_one()
        child_key = model.bad_child()
        queries_ = [parent_one_key]
        obs_ = {child_key: torch.tensor(0.0)}

        with self.assertRaises(RuntimeError):
            torch.manual_seed(17)
            hmc.infer(queries_, obs_, 10)
