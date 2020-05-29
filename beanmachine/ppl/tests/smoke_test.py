# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch.distributions as dist


class ToplevelSmokeTest(unittest.TestCase):
    def test_toplevel_package_imports(self):
        # these decorators should execute without error
        @bm.random_variable
        def foo(i):
            return dist.Bernoulli(0.5)

        @bm.functional
        def foo_sum(n):
            return sum(foo(i) for i in range(n))

        # exercise invocation from top-level package directly
        samples = bm.CompositionalInference().infer(
            [foo_sum(3)], {foo(0): False}, 1000, 1
        )
        bm.Diagnostics(samples)
