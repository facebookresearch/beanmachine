# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.legacy.inference import SingleSiteAncestralMetropolisHastings


class SamplerTest(unittest.TestCase):
    class SampleModel:
        @bm.random_variable
        def foo(self):
            return dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_sampler_smoke(self):
        model = self.SampleModel()
        num_samples = 10
        sampler = SingleSiteAncestralMetropolisHastings().sampler(
            [model.foo()],
            {model.bar(): torch.tensor(0.8)},
            num_samples,
            num_adaptive_samples=num_samples,
        )
        samples = []
        for sample in sampler:
            self.assertIn(model.foo(), sample)
            # only a single sample is returned at a time
            self.assertEqual(sample[model.foo()].numel(), 1)
            samples.append(sample)
        self.assertEqual(len(samples), num_samples)
        samples = sampler.to_monte_carlo_samples(samples)
        self.assertIn(model.foo(), samples)
        self.assertIsInstance(samples[model.foo()], torch.Tensor)
        self.assertEqual(samples[model.foo()].shape, (1, num_samples))

    def test_infinite_sampler(self):
        model = self.SampleModel()
        sampler = bm.SingleSiteRandomWalk().sampler(
            [model.foo()], {model.bar(): torch.tensor(0.4)}
        )
        for _ in range(10):
            sample = next(sampler)
            self.assertIn(model.foo(), sample)

    def test_multiple_samplers(self):
        model = self.SampleModel()
        num_chains = 2
        num_samples = 10
        samplers = [
            SingleSiteAncestralMetropolisHastings().sampler(
                [model.foo()], {model.bar(): torch.tensor(0.3)}, num_samples
            )
            for _ in range(num_chains)
        ]
        chains = [list(sampler) for sampler in samplers]
        samples = samplers[0].to_monte_carlo_samples(chains)
        self.assertIn(model.foo(), samples)
        self.assertEqual(samples[model.foo()].shape, (num_chains, num_samples))

    def test_thinning(self):
        mock_out = iter(range(20))

        def mock_single_iter(*args, **kwargs):
            return next(mock_out)

        model = self.SampleModel()
        sampler = SingleSiteAncestralMetropolisHastings().sampler(
            [model.foo()], {model.bar(): torch.tensor(0.3)}, thinning=4
        )
        # mock the result of inference
        sampler.kernel._single_iteration_run = mock_single_iter
        expected = 0
        for _ in range(5):
            out = next(sampler)
            self.assertEqual(out, expected)
            expected += 4
