# Copyright (c) Facebook, Inc. and its affiliates.
"""Parameterized test to compare samples from original
   and conjugate prior transformed models"""

import random
import unittest

import scipy
import torch
from beanmachine.ppl.compiler.testlib.conjugate_models import (
    BetaBernoulliBasicModel,
    BetaBernoulliOpsModel,
    BetaBernoulliScaleHyperParameters,
)
from beanmachine.ppl.inference.bmg_inference import BMGInference
from parameterized import parameterized, parameterized_class


_alpha = 2.0
_beta = 2.0

test_models = [
    (BetaBernoulliBasicModel, "BetaBernoulliConjugateFixer"),
    (BetaBernoulliOpsModel, "BetaBernoulliConjugateFixer"),
    (BetaBernoulliScaleHyperParameters, "BetaBernoulliConjugateFixer"),
]


def name_func(cls, _, params_dict):
    class_name = str(params_dict.get("model"))
    return f"{cls.__name__}_{parameterized.to_safe_name(class_name)}"


@parameterized_class(
    [{"model": test_model, "opt": opt} for test_model, opt in test_models],
    class_name_func=name_func,
)
class TestConjugacyTransformations(unittest.TestCase):
    def setUp(self):
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        num_samples = 1000
        num_obs = 4
        bmg = BMGInference()
        model = self.model(_alpha, _beta)
        observations = model.gen_obs(num_obs)
        queries = [model.theta()]

        # Generate samples from model when opt is disabled
        skip_optimizations = {self.opt}
        posterior_original = bmg.infer(queries, observations, num_samples)
        self.graph_original = bmg.to_dot(
            queries, observations, skip_optimizations=skip_optimizations
        )
        self.theta_samples_original = posterior_original[model.theta()][0]

        # Generate samples from model when opt is enabled
        skip_optimizations = set()
        posterior_transformed = bmg.infer(
            queries, observations, num_samples, 1, skip_optimizations=skip_optimizations
        )
        self.graph_transformed = bmg.to_dot(
            queries, observations, skip_optimizations=skip_optimizations
        )
        self.theta_samples_transformed = posterior_transformed[model.theta()][0]

    def test_samples_with_ks(self) -> None:
        self.assertNotEqual(
            self.graph_original.strip(),
            self.graph_transformed.strip(),
            "Original and transformed graph should not be identical.",
        )

        self.assertEqual(
            type(self.theta_samples_original),
            type(self.theta_samples_transformed),
            "Sample type of original and transformed model should be the same.",
        )

        self.assertEqual(
            len(self.theta_samples_original),
            len(self.theta_samples_transformed),
            "Sample size of original and transformed model should be the same.",
        )

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(
                self.theta_samples_original, self.theta_samples_transformed
            ).pvalue,
            0.05,
        )


if __name__ == "__main__":
    unittest.main()