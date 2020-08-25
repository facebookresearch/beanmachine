# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch.distributions as dist
import torch.tensor as tensor


class ToplevelSmokeTest(unittest.TestCase):
    def test_toplevel_package_imports(self):
        # these decorators should execute without error
        @bm.random_variable
        def foo(i):
            return dist.Bernoulli(0.5)

        @bm.functional
        def foo_sum(n):
            return sum(foo(i) for i in range(n))

        @bm.random_variable
        def bar():
            return dist.Normal(0, 1)

        # exercise invocation from top-level package directly
        # Compositional Inference
        samples = bm.CompositionalInference().infer(
            [foo_sum(3)], {foo(0): tensor(0.0)}, 1000, 1
        )
        bm.Diagnostics(samples)

        # Rejection Sampling
        samples = bm.RejectionSampling().infer([foo_sum(2)], {foo(0): False}, 1000, 1)
        bm.Diagnostics(samples)

        # NUTS
        samples = bm.SingleSiteNoUTurnSampler().infer(
            [bar()], {foo(0): tensor(0.0)}, 500, 1, num_adaptive_samples=500
        )
        bm.Diagnostics(samples)
