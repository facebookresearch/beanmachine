# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.abc.abc_infer import ApproximateBayesianComputation


class ApproximateBayesianComputationTest(unittest.TestCase):
    class CoinTossModel:
        def __init__(self, observation_shape):
            self.observation_shape = observation_shape

        @bm.random_variable
        def bias(self):
            return dist.Beta(0.5, 0.5)

        @bm.random_variable
        def coin_toss(self):
            return dist.Bernoulli(self.bias().repeat(self.observation_shape))

        def toss_head_count(self, toss_vals):
            return torch.sum(toss_vals)

        def toss_mean(self, toss_vals):
            return torch.mean(toss_vals)

        @bm.functional
        def num_heads(self):
            return self.toss_head_count(self.coin_toss())

        @bm.functional
        def mean_value(self):
            return self.toss_mean(self.coin_toss())

    def test_abc_inference(self):
        model = self.CoinTossModel(observation_shape=10)
        COIN_TOSS_DATA = dist.Bernoulli(0.9).sample([10])
        num_heads_key = model.num_heads()
        mean_value_key = model.mean_value()
        abc = ApproximateBayesianComputation(
            tolerance={num_heads_key: 1.0, mean_value_key: 0.1}
        )
        observations = {
            num_heads_key: model.toss_head_count(COIN_TOSS_DATA),
            mean_value_key: model.toss_mean(COIN_TOSS_DATA),
        }
        queries = [model.bias()]
        samples = abc.infer(
            queries, observations, num_samples=10, num_chains=1, verbose=None
        )
        mean = torch.mean(samples[model.bias()][0])
        self.assertTrue(mean.item() > 0.65)
        abc.reset()

    def test_abc_inference_with_singleton_arguments(self):
        model = self.CoinTossModel(observation_shape=10)
        COIN_TOSS_DATA = dist.Bernoulli(0.9).sample([10])
        abc = ApproximateBayesianComputation(
            distance_function=torch.dist, tolerance=1.0
        )
        observations = {
            model.num_heads(): model.toss_head_count(COIN_TOSS_DATA),
            model.mean_value(): model.toss_mean(COIN_TOSS_DATA),
        }
        queries = [model.bias()]
        samples = abc.infer(
            queries, observations, num_samples=10, num_chains=1, verbose=None
        )
        mean = torch.mean(samples[model.bias()][0])
        self.assertTrue(mean.item() > 0.65)
        abc.reset()

    def test_single_inference_step(self):
        model = self.CoinTossModel(observation_shape=10)
        abc = ApproximateBayesianComputation(tolerance={model.num_heads(): 1.0})
        abc.observations_ = {model.num_heads(): torch.tensor(15.0)}
        self.assertEqual(abc._single_inference_step(), 0.0)
        abc.reset()

    def test_max_attempts(self):
        model = self.CoinTossModel(observation_shape=100)
        COIN_TOSS_DATA = dist.Bernoulli(0.9).sample([100])
        abc = ApproximateBayesianComputation(
            tolerance={model.num_heads(): 0.1}, max_attempts_per_sample=2
        )
        observations = {model.num_heads(): model.toss_head_count(COIN_TOSS_DATA)}
        queries = [model.bias()]
        with self.assertRaises(RuntimeError):
            abc.infer(
                queries, observations, num_samples=100, num_chains=1, verbose=None
            )
        abc.reset()

    def test_shape_mismatch(self):
        model = self.CoinTossModel(observation_shape=100)
        abc = ApproximateBayesianComputation(tolerance={model.num_heads(): 0.1})
        observations = {model.num_heads(): torch.tensor([3, 4])}
        queries = [model.bias()]
        with self.assertRaises(ValueError):
            abc.infer(
                queries, observations, num_samples=100, num_chains=1, verbose=None
            )
        abc.reset()

    def test_simulate_mode(self):
        model = self.CoinTossModel(observation_shape=10)
        COIN_TOSS_DATA = dist.Bernoulli(0.9).sample([10])
        abc = ApproximateBayesianComputation(
            tolerance={model.num_heads(): 1, model.mean_value(): 0.1}
        )
        observations = {
            model.num_heads(): model.toss_head_count(COIN_TOSS_DATA),
            model.mean_value(): model.toss_mean(COIN_TOSS_DATA),
        }
        queries = [model.bias()]
        samples = abc.infer(
            queries, observations, num_samples=1, num_chains=1, verbose=None
        )
        # simulate 10 coin tosses from accepted bias sample
        sim_observations = {model.bias(): samples[model.bias()][0]}
        sim_queries = [model.coin_toss()]
        sim_abc = ApproximateBayesianComputation(simulate=True)
        sim_samples = sim_abc.infer(
            sim_queries, sim_observations, num_samples=10, num_chains=1, verbose=None
        )
        self.assertTrue(torch.sum(sim_samples[model.coin_toss()][0] == 1.0) > 5)
