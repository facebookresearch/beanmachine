# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest

import beanmachine.ppl as bm
import numpy as np
import torch
import torch.distributions as dist
import xarray as xr
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples, merge_dicts


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
        self.assertEqual(mcs.num_chains, 4)
        self.assertCountEqual(mcs.keys(), [foo_key])

        mcs = mh.infer([foo_key], {}, 7, num_adaptive_samples=3)

        self.assertEqual(mcs.num_adaptive_samples, 3)
        self.assertEqual(mcs[foo_key].shape, torch.zeros(4, 7).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(4, 7).shape)
        self.assertEqual(
            mcs.get_variable(foo_key, True).shape, torch.zeros(4, 10).shape
        )
        self.assertEqual(mcs.get_chain(3)[foo_key].shape, torch.zeros(7).shape)
        self.assertEqual(mcs.num_chains, 4)
        self.assertCountEqual(mcs.keys(), [foo_key])

    def test_one_chain(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        bar_key = model.bar()
        mcs = mh.infer([foo_key, bar_key], {}, 10, 1)

        self.assertEqual(mcs[foo_key].shape, torch.zeros(1, 10).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(1, 10).shape)
        self.assertEqual(mcs.get_chain()[foo_key].shape, torch.zeros(10).shape)
        self.assertEqual(mcs.num_chains, 1)
        self.assertCountEqual(mcs.keys(), [foo_key, bar_key])

        mcs = mh.infer([foo_key, bar_key], {}, 7, 1, num_adaptive_samples=3)

        self.assertEqual(mcs.num_adaptive_samples, 3)
        self.assertEqual(mcs[foo_key].shape, torch.zeros(1, 7).shape)
        self.assertEqual(mcs.get_variable(foo_key).shape, torch.zeros(1, 7).shape)
        self.assertEqual(
            mcs.get_variable(foo_key, True).shape, torch.zeros(1, 10).shape
        )
        self.assertEqual(mcs.get_chain()[foo_key].shape, torch.zeros(7).shape)
        self.assertEqual(mcs.num_chains, 1)
        self.assertCountEqual(mcs.keys(), [foo_key, bar_key])

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
            r" been restricted to a single chain",
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
        self.assertEqual(mcs.get_num_samples(), 10)
        self.assertEqual(mcs.get_num_samples(include_adapt_steps=True), 13)

    def test_dump_and_restore_samples(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        samples = mh.infer([foo_key], {}, num_samples=10, num_chains=2)
        self.assertEqual(samples[foo_key].shape, (2, 10))

        dumped = pickle.dumps((model, samples))
        # delete local variables and pretend that we are starting from a new session
        del model
        del mh
        del foo_key
        del samples

        # reload from dumped bytes
        reloaded_model, reloaded_samples = pickle.loads(dumped)
        # check the values still exist and have the correct shape
        self.assertEqual(reloaded_samples[reloaded_model.foo()].shape, (2, 10))

    def test_get_rv_with_default(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        samples = mh.infer([foo_key], {}, num_samples=10, num_chains=2)

        self.assertIn(model.foo(), samples)
        self.assertIsInstance(samples.get(model.foo()), torch.Tensor)
        self.assertIsNone(samples.get(model.bar()))
        self.assertEqual(samples.get(model.foo(), chain=0).shape, (10,))

    def test_merge_dicts(self):
        model = self.SampleModel()
        chain_lists = [{model.foo(): torch.rand(3)}, {model.foo(): torch.rand(3)}]
        rv_dict = merge_dicts(chain_lists)
        self.assertIn(model.foo(), rv_dict)
        self.assertEqual(rv_dict.get(model.foo()).shape, (2, 3))
        chain_lists.append({model.bar(): torch.rand(3)})
        with self.assertRaises(ValueError):
            merge_dicts(chain_lists)

    def test_type_conversion(self):
        model = self.SampleModel()
        samples = MonteCarloSamples(
            [{model.foo(): torch.rand(5), model.bar(): torch.rand(5)}],
            num_adaptive_samples=3,
        )

        xr_dataset = samples.to_xarray()
        self.assertIsInstance(xr_dataset, xr.Dataset)
        self.assertIn(model.foo(), xr_dataset)
        assert np.allclose(samples[model.bar()].numpy(), xr_dataset[model.bar()])
        xr_dataset = samples.to_xarray(include_adapt_steps=True)
        self.assertEqual(xr_dataset[model.foo()].shape, (1, 5))

        inference_data = samples.to_inference_data()
        self.assertIn(model.foo(), inference_data.posterior)

    def test_get_variable(self):
        model = self.SampleModel()
        samples = MonteCarloSamples(
            [{model.foo(): torch.arange(10)}], num_adaptive_samples=3
        ).get_chain(0)
        self.assertTrue(
            torch.all(samples.get_variable(model.foo()) == torch.arange(3, 10))
        )
        self.assertTrue(
            torch.all(samples.get_variable(model.foo(), True) == torch.arange(10))
        )

    def test_get_log_likehoods(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        bar_key = model.bar()
        mcs = mh.infer(
            [foo_key],
            {bar_key: torch.tensor(4.0)},
            num_samples=5,
            num_chains=2,
        )
        self.assertTrue(hasattr(mcs, "log_likelihoods"))
        self.assertIn(bar_key, mcs.log_likelihoods)
        self.assertTrue(hasattr(mcs, "adaptive_log_likelihoods"))
        self.assertIn(bar_key, mcs.adaptive_log_likelihoods)
        self.assertEqual(
            mcs.get_log_likelihoods(bar_key).shape, torch.zeros(2, 5).shape
        )
        mcs = mcs.get_chain(0)
        self.assertEqual(mcs.get_log_likelihoods(bar_key).shape, torch.zeros(5).shape)

        mcs = mh.infer(
            [foo_key],
            {bar_key: torch.tensor(4.0)},
            num_samples=5,
            num_chains=2,
            num_adaptive_samples=3,
        )

        self.assertEqual(
            mcs.get_log_likelihoods(bar_key).shape, torch.zeros(2, 5).shape
        )
        self.assertEqual(
            mcs.adaptive_log_likelihoods[bar_key].shape, torch.zeros(2, 3).shape
        )
        self.assertEqual(
            mcs.get_chain(0).get_log_likelihoods(bar_key).shape, torch.zeros(5).shape
        )
        self.assertEqual(
            mcs.get_log_likelihoods(bar_key, True).shape, torch.zeros(2, 8).shape
        )
        self.assertEqual(
            mcs.get_chain(0).adaptive_log_likelihoods[bar_key].shape,
            torch.zeros(1, 3).shape,
        )

    def test_thinning(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        samples = mh.infer([model.foo()], {}, num_samples=20, num_chains=1)

        self.assertEqual(samples.get(model.foo(), chain=0).shape, (20,))
        self.assertEqual(samples.get(model.foo(), chain=0, thinning=4).shape, (5,))

    def test_to_inference_data(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        samples = mh.infer([model.foo()], {}, num_samples=10, num_chains=1)
        az_xarray = samples.to_inference_data()
        self.assertNotIn("warmup_posterior", az_xarray)

        samples = mh.infer(
            [model.foo()], {}, num_samples=10, num_adaptive_samples=2, num_chains=1
        )
        az_xarray = samples.to_inference_data(include_adapt_steps=True)
        self.assertIn("warmup_posterior", az_xarray)
