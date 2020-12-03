# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_wrapper import RVWrapper
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import World


# used in testing global random variable
@bm.random_variable
def foo():
    return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))


class RVWrapperTest(unittest.TestCase):
    class SampleModel:
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

        @bm.random_variable
        def baz(self, i: int):
            return dist.Normal(self.foo(), self.bar())

        def demo(self):
            pass

    def test_rv_wrapper_type(self):
        model = self.SampleModel()

        self.assertIsInstance(model.foo, RVWrapper)
        self.assertIsInstance(model.bar, RVWrapper)
        self.assertIsInstance(foo, RVWrapper)
        self.assertIsInstance(self.SampleModel.bar, RVWrapper)

    def test_rv_wrapper_instance_binding(self):
        m = self.SampleModel()

        self.assertFalse(inspect.ismethod(foo.function))
        self.assertFalse(inspect.ismethod(self.SampleModel.foo.function))
        self.assertTrue(inspect.ismethod(m.foo.function))

        # check if m.foo.function is correctly bound to model
        self.assertIs(m.foo.function.__self__, m)
        self.assertIs(m.foo.model, m)  # this is equivalent to the line above
        # method.__func__ should refer to the underlying (unbound) function
        self.assertIs(m.foo.function.__func__, self.SampleModel.foo.function)

        # check if global random variable is indeed unbound
        self.assertIsNone(foo.model)

    def test_calling_rv_wrapper(self):
        world = World()
        model = self.SampleModel()

        baz_key = model.baz(1)
        self.assertIsInstance(baz_key, RVIdentifier)

        self.assertIsInstance(baz_key.wrapper, RVWrapper)
        self.assertEqual(baz_key.wrapper, model.baz)
        # Notice that when we call __get__, a new instance of RVWrapper is created,
        # which might be at a different address (so we should check for value
        # equivalence rather than address equivalence), i.e.
        self.assertIsNot(baz_key.wrapper, model.baz)
        # in fact, even on regular Python function, binding the same function twice
        # will create two methods object
        self.assertIsNot(model.demo, model.demo)

        # RVIdentifier should store a tuple of arguments (excluding 'self')
        self.assertTupleEqual(baz_key.arguments, (1,))

        with world:
            # the following calls should be equivalent
            value1 = model.baz(1)
            value2 = baz_key.wrapper(*baz_key.arguments)
        self.assertEqual(value1, value2)

        # the same applies for global random variable
        foo_key = foo()
        with world:
            value1 = foo()
            value2 = foo_key.wrapper(*foo_key.arguments)
        self.assertEqual(value1, value2)
