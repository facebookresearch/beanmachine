# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from contextlib import closing
from io import BytesIO

import beanmachine.ppl as bm
import dill
import torch
import torch.distributions as dist


class MonteCarloSamplesTest(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_default_four_chains(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        mcs = mh.infer([foo_key], {}, 10)

        self.assertEqual(mcs[foo_key].shape, torch.zeros(4, 10).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(4, 10).shape)
        self.assertEqual(mcs.get_chain(3)[foo_key].shape, torch.zeros(10).shape)
        self.assertEqual(mcs.get_num_chains(), 4)
        self.assertEqual(mcs.get_rv_names(), [foo_key])

        mcs.num_adaptive_samples = 3

        self.assertEqual(mcs[foo_key].shape, torch.zeros(4, 7).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(4, 7).shape)
        self.assertEqual(
            mcs.get_variable(foo_key, True).shape, torch.zeros(4, 10).shape
        )
        self.assertEqual(mcs.get_chain(3)[foo_key].shape, torch.zeros(7).shape)
        self.assertEqual(mcs.get_num_chains(), 4)
        self.assertEqual(mcs.get_rv_names(), [foo_key])

    def test_one_chain(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        bar_key = model.bar()
        mcs = mh.infer([foo_key, bar_key], {}, 10, 1)

        self.assertEqual(mcs[foo_key].shape, torch.zeros(1, 10).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(1, 10).shape)
        self.assertEqual(mcs.get_chain()[foo_key].shape, torch.zeros(10).shape)
        self.assertEqual(mcs.get_num_chains(), 1)
        self.assertEqual(mcs.get_rv_names(), [foo_key, bar_key])

        mcs.num_adaptive_samples = 3

        self.assertEqual(mcs[foo_key].shape, torch.zeros(1, 7).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(1, 7).shape)
        self.assertEqual(
            mcs.get_variable(foo_key, True).shape, torch.zeros(1, 10).shape
        )
        self.assertEqual(mcs.get_chain()[foo_key].shape, torch.zeros(7).shape)
        self.assertEqual(mcs.get_num_chains(), 1)
        self.assertEqual(mcs.get_rv_names(), [foo_key, bar_key])

    def test_chain_exceptions(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        mcs = mh.infer([foo_key], {}, 10)

        with self.assertRaisesRegex(IndexError, r"Please specify a valid chain"):
            mcs.get_chain(-1)

        with self.assertRaisesRegex(IndexError, r"Please specify a valid chain"):
            mcs.get_chain(4)

        with self.assertRaisesRegex(
            ValueError,
            r"The current MonteCarloSamples object has already"
            r" been restricted to chain \d+",
        ):
            one_chain = mcs.get_chain()
            one_chain.get_chain()

    def test_num_adaptive_samples(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        mcs = mh.infer([foo_key], {}, 10, num_adaptive_samples=3)

        self.assertEqual(mcs[foo_key].shape, torch.zeros(4, 10).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(4, 10).shape)
        self.assertEqual(
            mcs.get_variable(foo_key, include_adapt_steps=True).shape,
            torch.zeros(4, 13).shape,
        )

    def test_serializable(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        mcs = mh.infer([model.foo()], {}, 10, 1)

        with closing(BytesIO()) as fp:
            # should SerDe without exception
            dill.dump((mcs, model), fp)
            fp.seek(0)
            samples_deser, model_deser = dill.load(fp)

            # should be able to lookup using deserialized instance methods
            self.assertTrue(model_deser.foo() in samples_deser.data.rv_dict)

            # NOTE: samples_deser is keyed on model_deser.foo(), NOT model.foo()
            self.assertTrue(model.foo() not in samples_deser.data.rv_dict)
