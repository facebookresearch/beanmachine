# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.rejection_sampling_infer import RejectionSampling


class RejectionSamplingTest(unittest.TestCase):
    class SampleModel:
        @bm.random_variable
        def foo(self):
            return dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Bernoulli(self.foo())

    class SampleFunctionalModel:
        @bm.random_variable
        def foo(self):
            return dist.Categorical(probs=torch.ones(7) / 7.0)

        @bm.functional
        def bar(self):
            return torch.tensor(15.0) + self.foo()

    class VectorModel:
        @bm.random_variable
        def foo(self):
            return dist.Beta(0.25, 0.25)

        @bm.random_variable
        def bar(self):
            return dist.Bernoulli(self.foo().repeat([3]))

    class RealValuedVectorModel:
        @bm.random_variable
        def foo(self):
            return dist.Uniform(0, 1)

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo().repeat([3]), 0.1)

    def test_rejection_sampling(self):
        model = self.SampleModel()
        bar_key = model.bar()
        observations = {bar_key: torch.tensor(1.0)}
        rej = RejectionSampling()
        num_samples = 1000
        samples = rej.infer(
            queries=[model.foo()],
            observations=observations,
            num_samples=num_samples,
            num_chains=1,
        )
        mean = torch.mean(samples[model.foo()][0])
        self.assertTrue(mean > 0.6)

    def test_single_inference_step(self):
        model = self.SampleModel()
        bar_key = model.bar()
        rej = RejectionSampling()
        rej.observations_ = {bar_key: torch.tensor(2)}
        self.assertEqual(rej._single_inference_step(), 0)
        rej.reset()

    def test_inference_over_functionals(self):
        model = self.SampleFunctionalModel()
        rej = RejectionSampling()
        queries = [model.foo()]
        observations = {model.bar(): torch.tensor(20).int()}
        samples = rej.infer(queries, observations, num_samples=100, num_chains=1)
        self.assertTrue((samples[model.foo()][0] == 5).all())

    def test_vectorized_inference(self):
        model = self.VectorModel()
        bar_key = model.bar()
        foo_key = model.foo()
        bar_observations = {bar_key: torch.ones(3)}
        num_samples = 1000
        rej = RejectionSampling()
        samples = rej.infer([foo_key], bar_observations, num_samples, 1)
        mean = torch.mean(samples[model.foo()][0])
        self.assertTrue(mean.item() > 0.75)

    def test_max_attempts(self):
        model = self.SampleModel()
        bar_key = model.bar()
        observations = {bar_key: torch.tensor(1.0)}
        rej = RejectionSampling(max_attempts_per_sample=5)
        num_samples = 1000
        with self.assertRaises(RuntimeError):
            rej.infer(
                queries=[model.foo()],
                observations=observations,
                num_samples=num_samples,
                num_chains=1,
            )
        rej.reset()

    def test_rejection_with_tolerance(self):
        model = self.RealValuedVectorModel()
        bar_key = model.bar()
        observations = {bar_key: torch.tensor([0.1551, 0.1550, 0.1552])}
        rej = RejectionSampling(tolerance=0.2)
        num_samples = 500
        samples = rej.infer(
            queries=[model.foo()],
            observations=observations,
            num_samples=num_samples,
            num_chains=1,
        )
        mean = torch.mean(samples[model.foo()][0])
        self.assertAlmostEqual(mean.item(), 0.15, delta=0.2)

    def test_shape_mismatch(self):
        model = self.RealValuedVectorModel()
        bar_key = model.bar()
        observations = {bar_key: torch.tensor([0.15, 0.155])}
        rej = RejectionSampling(tolerance=0.1)
        num_samples = 500
        with self.assertRaises(ValueError):
            rej.infer(
                queries=[model.foo()],
                observations=observations,
                num_samples=num_samples,
                num_chains=1,
            )
        rej.reset()
