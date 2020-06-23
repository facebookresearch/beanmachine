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

    def test_inference_over_functionals(self):
        model = self.SampleFunctionalModel()
        rej = RejectionSampling()
        queries = [model.foo()]
        observations = {model.bar(): torch.tensor(20).int()}
        samples = rej.infer(queries, observations, num_samples=100, num_chains=1)
        self.assertTrue((samples[model.foo()][0] == 5).all())

    def test_vectorized_inference(self):
        model = self.VectorModel()
        bar_observations = {model.bar(): torch.ones(3)}
        num_samples = 1000
        rej = RejectionSampling()
        samples = rej.infer([model.foo()], bar_observations, num_samples, 1)
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
