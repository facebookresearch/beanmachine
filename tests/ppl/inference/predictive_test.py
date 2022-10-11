# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
import torch
from beanmachine.ppl.examples.hierarchical_models import UniformBernoulliModel
from torch import tensor

from ..utils.fixtures import (
    approx_all,
    parametrize_inference,
    parametrize_model,
    parametrize_model_value,
    parametrize_value,
)


@parametrize_model([UniformBernoulliModel(tensor(0.0), tensor(1.0))])
class TestPredictive:
    @staticmethod
    def test_prior_predictive(model):
        queries = [model.prior(), model.likelihood()]
        predictives = bm.simulate(queries, num_samples=10)
        assert predictives[model.prior()].shape == (1, 10)
        assert predictives[model.likelihood()].shape == (1, 10)

    @staticmethod
    @parametrize_value([tensor([1.0, 0.0])])
    @pytest.mark.parametrize("num_chains", [2])
    @parametrize_inference([bm.SingleSiteAncestralMetropolisHastings()])
    @pytest.mark.parametrize("vectorized", [True, False])
    def test_posterior_predictive(model, value, inference, num_chains, vectorized):
        num_samples = 10
        shape_samples = (num_chains, num_samples) + model.lo.shape

        obs = {model.likelihood_i(i): value[i] for i in range(len(value))}
        post_samples = inference.infer(
            [model.prior()], obs, num_samples=num_samples, num_chains=num_chains
        )
        assert post_samples[model.prior()].shape == shape_samples

        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=vectorized)
        assert all(predictives[rv].shape == shape_samples for rv in obs.keys())

    @staticmethod
    def test_predictive_dynamic(model):
        obs = {
            model.likelihood_dynamic(0): torch.tensor([0.9]),
            model.likelihood_dynamic(1): torch.tensor([4.9]),
        }
        # only query one of the variables
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [model.prior()], obs, num_samples=10, num_chains=2
        )
        assert post_samples[model.prior()].shape == (2, 10)
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=False)
        assert predictives[model.likelihood_dynamic(0)].shape == (2, 10)
        assert predictives[model.likelihood_dynamic(1)].shape == (2, 10)

    @staticmethod
    def test_predictive_data(model):
        x = torch.randn(4)
        y = torch.randn(4) + 2.0
        obs = {model.likelihood_reg(x): y}
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [model.prior()], obs, num_samples=10, num_chains=2
        )
        assert post_samples[model.prior()].shape == (2, 10)
        test_x = torch.randn(4, 1, 1)
        test_query = model.likelihood_reg(test_x)
        predictives = bm.simulate([test_query], post_samples, vectorized=True)
        assert predictives[test_query].shape == (4, 2, 10)

    @staticmethod
    def test_empirical(model):
        obs = {
            model.likelihood_i(0): torch.tensor(1.0),
            model.likelihood_i(1): torch.tensor(0.0),
            model.likelihood_i(2): torch.tensor(0.0),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [model.prior()], obs, num_samples=10, num_chains=4
        )
        empirical = bm.empirical([model.prior()], post_samples, num_samples=26)
        assert empirical[model.prior()].shape == (1, 26)
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=True)
        empirical = bm.empirical(list(obs.keys()), predictives, num_samples=27)
        assert len(empirical) == 3
        assert empirical[model.likelihood_i(0)].shape == (1, 27)
        assert empirical[model.likelihood_i(1)].shape == (1, 27)

    @staticmethod
    def test_posterior_dict(model):
        obs = {
            model.likelihood_i(0): torch.tensor(1.0),
            model.likelihood_i(1): torch.tensor(0.0),
        }

        posterior = {model.prior(): torch.tensor([0.5, 0.5])}

        predictives_dict = bm.simulate(list(obs.keys()), posterior)
        assert predictives_dict[model.likelihood_i(0)].shape == (1, 2)
        assert predictives_dict[model.likelihood_i(1)].shape == (1, 2)

    @staticmethod
    def test_posterior_dict_predictive(model):
        obs = {
            model.likelihood_i(0): torch.tensor(1.0),
            model.likelihood_i(1): torch.tensor(0.0),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [model.prior()], obs, num_samples=10, num_chains=1
        )
        assert post_samples[model.prior()].shape == (1, 10)

        post_samples_dict = dict(post_samples)
        predictives_dict = bm.simulate(list(obs.keys()), post_samples_dict)
        assert predictives_dict[model.likelihood_i(0)].shape == (1, 10)
        assert predictives_dict[model.likelihood_i(1)].shape == (1, 10)


@parametrize_model_value(
    [
        (UniformBernoulliModel(tensor([0.0]), tensor([1.0])), tensor([1.0])),
        (
            UniformBernoulliModel(torch.zeros(1, 2), torch.ones(1, 2)),
            tensor([[[1.0, 1.0]], [[0.0, 1.0]]]),
        ),
    ]
)
@pytest.mark.parametrize("num_chains", [1, 3])
@parametrize_inference([bm.SingleSiteAncestralMetropolisHastings()])
class TestPredictiveMV:
    @staticmethod
    def test_posterior_predictive(model, value, inference, num_chains):
        torch.manual_seed(10)
        num_samples = 10
        shape_samples = (num_chains, num_samples) + model.lo.shape

        # define observations
        if value.ndim == model.lo.ndim:
            obs = {model.likelihood(): value}
        else:
            obs = {model.likelihood_i(i): value[i] for i in range(len(value))}

        # run inference
        post_samples = inference.infer(
            [model.prior()], obs, num_samples=num_samples, num_chains=num_chains
        )
        assert post_samples[model.prior()].shape == shape_samples

        # simulate predictives
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=True)
        for rv in obs.keys():
            assert predictives[rv].shape == shape_samples
        if value.ndim == 1 + model.lo.ndim:
            rvs = list(obs.keys())[:2]
            assert not approx_all(predictives[rvs[0]], predictives[rvs[1]], 0.5)
        inf_data = predictives.to_inference_data()
        result_keys = [
            "posterior",
            "observed_data",
            "log_likelihood",
            "posterior_predictive",
        ]
        for k in result_keys:
            assert k in inf_data
        for rv in obs.keys():
            assert inf_data.posterior_predictive[rv].shape == shape_samples
