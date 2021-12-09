# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist


@bm.random_variable
def foo():
    return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))


class RVIdentifierTest(unittest.TestCase):
    class SampleModel:
        @staticmethod
        @bm.random_variable
        def foo():
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self, sigma: float):
            return dist.Normal(self.foo(), torch.tensor(sigma))

        @bm.random_variable
        def baz(self):
            return dist.Normal(self.foo(), self.bar(1.0))

    class SampleModelWithEq:
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        def __eq__(self, other):
            return isinstance(other, RVIdentifierTest.SampleModelWithEq)

    def test_pickle_unbound_rv_identifier(self):
        original_foo_key = foo()
        foo_bytes = pickle.dumps(foo())
        reloaded_foo_key = pickle.loads(foo_bytes)

        # reloaded RVIdentifier should be equivalent to the original copy
        self.assertEqual(original_foo_key, reloaded_foo_key)
        self.assertEqual(reloaded_foo_key, foo())
        # In fact, when unpickling, it will recover the reference to the decorated
        # function
        self.assertIs(reloaded_foo_key.wrapper, foo)
        # ^ this requires the function to be available when unpickling

    def test_pickle_rv_with_same_name(self):
        rv_bytes = pickle.dumps((foo(), self.SampleModel.foo()))
        foo_key_1, foo_key_2 = pickle.loads(rv_bytes)

        self.assertEqual(foo(), foo_key_1)
        self.assertEqual(self.SampleModel.foo(), foo_key_2)
        # the two 'foo' functions with same name are not equivalent
        self.assertNotEqual(foo_key_1, foo_key_2)

    def test_pickle_bound_rv_identifier(self):
        model = self.SampleModel()
        bar_key = model.bar(3.0)

        # we should dump the model and RVIdentifier together if we want to recover the
        # reference
        model_and_rv_bytes = pickle.dumps((model, bar_key))
        reloaded_model, reloaded_bar_key = pickle.loads(model_and_rv_bytes)

        # We should be able to use the reloaded model to generate new RVIdentifier that
        # are equivalent to the unpickled ones
        self.assertEqual(reloaded_model.bar(3.0), reloaded_bar_key)
        # However, notice that the reloaded model is a copy of the original model with
        # the same value, so unless __eq__ is defined on the model, Python will compare
        # object by address (so the reloaded model & identifier are not equal to the
        # original ones)
        self.assertNotEqual(reloaded_model, model)
        self.assertNotEqual(bar_key, reloaded_bar_key)

    def test_pickle_bound_rv_in_model_with_eq_operator(self):
        model = self.SampleModelWithEq()
        foo_key = model.foo()
        model_and_rv_bytes = pickle.dumps((model, foo_key))
        reloaded_model, reloaded_foo_key = pickle.loads(model_and_rv_bytes)

        self.assertEqual(reloaded_model, model)
        self.assertEqual(foo_key, reloaded_foo_key)
        self.assertEqual(model.foo(), reloaded_foo_key)

        # Though instead of defining __eq__ and maintain multiple copies of the model,
        # it might be better to just use the unpickled model in a new session, i.e.
        del model  # mock the case where model is not defined in the new session yet
        model, bar_key = pickle.loads(model_and_rv_bytes)
        self.assertEqual(model.foo(), foo_key)

        # For global scope random variables, the definition of functions have to be
        # available when unpickling. Similarly, for class cope random variables, the
        # definition of class also needs to be available.

    def test_pickle_multiple_models(self):
        model1 = self.SampleModel()
        model2 = self.SampleModel()

        self.assertNotEqual(model1.baz(), model2.baz())
        rv_set = {model1.baz(), model2.baz(), model2.bar(1.5)}
        # the following will be similar to how
        serialized_bytes = pickle.dumps(
            {"model1": model1, "model2": model2, "values_to_keep": rv_set}
        )
        # notice that we can also dump the two models separately as long as they don't
        # cross reference each other

        # delete current variables and "start a new session"
        del model1
        del model2
        del rv_set

        restored_state = pickle.loads(serialized_bytes)
        model1 = restored_state.get("model1")
        model2 = restored_state.get("model2")
        rv_set = restored_state.get("values_to_keep")

        self.assertNotEqual(model1.baz(), model2.baz())
        self.assertIn(model1.baz(), rv_set)
        self.assertIn(model2.baz(), rv_set)
        self.assertNotIn(model1.bar(1.5), rv_set)
        self.assertIn(model2.bar(1.5), rv_set)
