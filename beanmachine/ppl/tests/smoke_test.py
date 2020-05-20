# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bmp
import torch.distributions as dist


class ToplevelSmokeTest(unittest.TestCase):
    def test_toplevel_package_imports(self):
        # these decorators should execute without error
        @bmp.random_variable
        def foo(i):
            return dist.Bernoulli(0.5)

        @bmp.functional
        def foo_sum(n):
            return sum(foo(i) for i in range(n))
