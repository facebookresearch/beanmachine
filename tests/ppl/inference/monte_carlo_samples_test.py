# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import beanmachine.ppl as bm
import numpy as np
import pytest
import torch
import xarray as xr
from beanmachine.ppl.examples.conjugate_models import NormalNormalModel
from beanmachine.ppl.inference.monte_carlo_samples import merge_dicts, MonteCarloSamples
from torch import tensor

from ..utils.fixtures import parametrize_inference, parametrize_model


pytestmark = parametrize_model(
    [NormalNormalModel(tensor(0.0), tensor(1.0), tensor(1.0))]
)


def test_merge_dicts(model):
    chain_lists = [{model.theta(): torch.rand(3)}, {model.theta(): torch.rand(3)}]
    rv_dict = merge_dicts(chain_lists)
    assert model.theta() in rv_dict
    assert rv_dict.get(model.theta()).shape == (2, 3)
    chain_lists.append({model.x(): torch.rand(3)})
    with pytest.raises(ValueError):
        merge_dicts(chain_lists)


def test_type_conversion(model):
    samples = MonteCarloSamples(
        [{model.theta(): torch.rand(5), model.x(): torch.rand(5)}],
        num_adaptive_samples=3,
    )

    xr_dataset = samples.to_xarray()
    assert isinstance(xr_dataset, xr.Dataset)
    assert model.theta() in xr_dataset
    assert np.allclose(samples[model.x()].numpy(), xr_dataset[model.x()])
    xr_dataset = samples.to_xarray(include_adapt_steps=True)
    assert xr_dataset[model.theta()].shape == (1, 5)

    inference_data = samples.to_inference_data()
    assert model.theta() in inference_data.posterior


def test_get_variable(model):
    samples = MonteCarloSamples(
        [{model.x(): torch.arange(10)}], num_adaptive_samples=3
    ).get_chain(0)
    assert torch.all(samples.get_variable(model.x()) == torch.arange(3, 10))
    assert torch.all(samples.get_variable(model.x(), True) == torch.arange(10))


@parametrize_inference([bm.SingleSiteAncestralMetropolisHastings()])
class TestInferenceResults:
    @staticmethod
    def test_default_four_chains(model, inference):
        p_key = model.theta()
        mcs = inference.infer([p_key], {}, 10)

        assert mcs[p_key].shape == torch.zeros(4, 10).shape
        assert mcs.get_variable(p_key).shape == torch.zeros(4, 10).shape
        assert mcs.get_chain(3)[p_key].shape == torch.zeros(10).shape
        assert mcs.num_chains == 4
        assert set(mcs.keys()) == set([p_key])

        mcs = inference.infer([p_key], {}, 7, num_adaptive_samples=3)

        assert mcs.num_adaptive_samples == 3
        assert mcs[p_key].shape == torch.zeros(4, 7).shape
        assert mcs.get_variable(p_key).shape == torch.zeros(4, 7).shape
        assert mcs.get_variable(p_key, True).shape == torch.zeros(4, 10).shape
        assert mcs.get_chain(3)[p_key].shape == torch.zeros(7).shape
        assert mcs.num_chains == 4
        assert set(mcs.keys()) == set([p_key])

    @staticmethod
    def test_one_chain(model, inference):
        p_key = model.theta()
        l_key = model.x()
        mcs = inference.infer([p_key, l_key], {}, 10, 1)

        assert mcs[p_key].shape == torch.zeros(1, 10).shape
        assert mcs.get_variable(p_key).shape == torch.zeros(1, 10).shape
        assert mcs.get_chain()[p_key].shape == torch.zeros(10).shape
        assert mcs.num_chains == 1
        assert set(mcs.keys()) == set([p_key, l_key])

        mcs = inference.infer([p_key, l_key], {}, 7, 1, num_adaptive_samples=3)

        assert mcs.num_adaptive_samples == 3
        assert mcs[p_key].shape == torch.zeros(1, 7).shape
        assert mcs.get_variable(p_key).shape == torch.zeros(1, 7).shape
        assert mcs.get_variable(p_key, True).shape == torch.zeros(1, 10).shape
        assert mcs.get_chain()[p_key].shape == torch.zeros(7).shape
        assert mcs.num_chains == 1
        assert set(mcs.keys()) == set([p_key, l_key])

    @staticmethod
    def test_chain_exceptions(model, inference):
        p_key = model.theta()
        mcs = inference.infer([p_key], {}, 10)

        with pytest.raises(IndexError, match="Please specify a valid chain"):
            mcs.get_chain(-1)
        with pytest.raises(IndexError, match="Please specify a valid chain"):
            mcs.get_chain(4)
        with pytest.raises(
            ValueError,
            match=(
                r"The current MonteCarloSamples object has already"
                r" been restricted to a single chain"
            ),
        ):
            one_chain = mcs.get_chain()
            one_chain.get_chain()

    @staticmethod
    def test_num_adaptive_samples(model, inference):
        p_key = model.theta()
        mcs = inference.infer([p_key], {}, 10, num_adaptive_samples=3)

        assert mcs[p_key].shape == torch.zeros(4, 10).shape
        assert mcs.get_variable(p_key).shape == torch.zeros(4, 10).shape
        assert (
            mcs.get_variable(p_key, include_adapt_steps=True).shape
            == torch.zeros(4, 13).shape
        )
        assert mcs.get_num_samples() == 10
        assert mcs.get_num_samples(include_adapt_steps=True) == 13

    @staticmethod
    def test_dump_and_restore_samples(model, inference):
        p_key = model.theta()
        samples = inference.infer([p_key], {}, num_samples=10, num_chains=2)
        assert samples[p_key].shape == (2, 10)

        dumped = pickle.dumps((model, samples))
        # delete local variables and pretend that we are starting from a new session
        del model
        del inference
        del p_key
        del samples

        # reload from dumped bytes
        reloaded_model, reloaded_samples = pickle.loads(dumped)
        # check the values still exist and have the correct shape
        assert reloaded_samples[reloaded_model.theta()].shape == (2, 10)

    @staticmethod
    def test_get_rv_with_default(model, inference):
        p_key = model.theta()
        samples = inference.infer([p_key], {}, num_samples=10, num_chains=2)

        assert model.theta() in samples
        assert isinstance(samples.get(model.theta()), torch.Tensor)
        assert samples.get(model.x()) is None
        assert samples.get(model.theta(), chain=0).shape == (10,)

    @staticmethod
    def test_get_log_likehoods(model, inference):
        p_key = model.theta()
        l_key = model.x()
        mcs = inference.infer(
            [p_key],
            {l_key: torch.tensor(4.0)},
            num_samples=5,
            num_chains=2,
        )
        assert hasattr(mcs, "log_likelihoods")
        assert l_key in mcs.log_likelihoods
        assert hasattr(mcs, "adaptive_log_likelihoods")
        assert l_key in mcs.adaptive_log_likelihoods
        assert mcs.get_log_likelihoods(l_key).shape == torch.zeros(2, 5).shape
        mcs = mcs.get_chain(0)
        assert mcs.get_log_likelihoods(l_key).shape == torch.zeros(5).shape

        mcs = inference.infer(
            [p_key],
            {l_key: torch.tensor(4.0)},
            num_samples=5,
            num_chains=2,
            num_adaptive_samples=3,
        )
        assert mcs.get_log_likelihoods(l_key).shape == torch.zeros(2, 5).shape
        assert mcs.adaptive_log_likelihoods[l_key].shape == torch.zeros(2, 3).shape
        assert mcs.get_chain(0).get_log_likelihoods(l_key).shape == torch.zeros(5).shape
        assert mcs.get_log_likelihoods(l_key, True).shape == torch.zeros(2, 8).shape
        assert (
            mcs.get_chain(0).adaptive_log_likelihoods[l_key].shape
            == torch.zeros(1, 3).shape
        )

    @staticmethod
    def test_thinning(model, inference):
        samples = inference.infer([model.theta()], {}, num_samples=20, num_chains=1)
        assert samples.get(model.theta(), chain=0).shape == (20,)
        assert samples.get(model.theta(), chain=0, thinning=4).shape == (5,)

    @staticmethod
    def test_add_group(model, inference):
        samples = inference.infer([model.theta()], {}, num_samples=20, num_chains=1)
        new_samples = MonteCarloSamples(samples.samples, default_namespace="new")
        new_samples.add_groups(samples)
        assert samples.observations == new_samples.observations
        assert samples.log_likelihoods == new_samples.log_likelihoods
        assert "posterior" in new_samples.namespaces

    @staticmethod
    def test_to_inference_data(model, inference):
        samples = inference.infer([model.theta()], {}, num_samples=10, num_chains=1)
        az_xarray = samples.to_inference_data()
        assert "warmup_posterior" not in az_xarray

        samples = inference.infer(
            [model.theta()], {}, num_samples=10, num_adaptive_samples=2, num_chains=1
        )
        az_xarray = samples.to_inference_data(include_adapt_steps=True)
        assert "warmup_posterior" in az_xarray
