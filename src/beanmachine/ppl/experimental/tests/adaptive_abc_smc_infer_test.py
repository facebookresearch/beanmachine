# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.abc.adaptive_abc_smc_infer import (
    AdaptiveApproximateBayesianComputationSequentialMonteCarlo,
)


class AdaptiveApproximateBayesianComputationSequentialMonteCarloTest(unittest.TestCase):
    torch.manual_seed(42)

    class CoinTossModel:
        def __init__(self, observation_shape):
            self.observation_shape = observation_shape

        @bm.random_variable
        def bias(self):
            return dist.Beta(0.5, 0.5)

        @bm.random_variable
        def coin_toss(self):
            return dist.Bernoulli(self.bias().repeat(self.observation_shape))

        def toss_mean(self, toss_vals):
            return torch.mean(toss_vals)

        @bm.functional
        def mean_value(self):
            return self.toss_mean(self.coin_toss())

    def test_adaptive_abc_smc_inference(self):
        model = self.CoinTossModel(observation_shape=10)
        COIN_TOSS_DATA = dist.Bernoulli(0.77).sample([10])
        mean_value_key = model.mean_value()
        adaptive_abc_smc = AdaptiveApproximateBayesianComputationSequentialMonteCarlo(
            initial_tolerance=0.5, target_tolerance=0.1
        )
        observations = {mean_value_key: model.toss_mean(COIN_TOSS_DATA)}
        queries = [model.bias()]
        samples = adaptive_abc_smc.infer(
            queries, observations, num_samples=100, num_chains=1
        )
        self.assertAlmostEqual(
            torch.mean(samples[model.bias()][0]).item(), 0.77, delta=0.3
        )
        adaptive_abc_smc.reset()

    def test_max_attempts(self):
        model = self.CoinTossModel(observation_shape=100)
        COIN_TOSS_DATA = dist.Bernoulli(0.9).sample([100])
        mean_value_key = model.mean_value()
        adaptive_abc_smc = AdaptiveApproximateBayesianComputationSequentialMonteCarlo(
            initial_tolerance=0.5, target_tolerance=0.1, max_attempts_per_sample=2
        )
        observations = {mean_value_key: model.toss_mean(COIN_TOSS_DATA)}
        queries = [model.bias()]
        with self.assertRaises(RuntimeError):
            adaptive_abc_smc.infer(
                queries, observations, num_samples=100, num_chains=1, verbose=None
            )
        adaptive_abc_smc.reset()

    def test_shape_mismatch(self):
        model = self.CoinTossModel(observation_shape=100)
        adaptive_abc_smc = AdaptiveApproximateBayesianComputationSequentialMonteCarlo(
            initial_tolerance=0.5, target_tolerance=0.1
        )
        observations = {model.mean_value(): torch.tensor([3, 4])}
        queries = [model.bias()]
        with self.assertRaises(ValueError):
            adaptive_abc_smc.infer(
                queries, observations, num_samples=100, num_chains=1, verbose=None
            )
        adaptive_abc_smc.reset()
