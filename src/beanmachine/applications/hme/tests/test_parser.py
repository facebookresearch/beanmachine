# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import pytest
from beanmachine.applications.hme import (
    HME,
    ModelConfig,
    RegressionConfig,
)
from beanmachine.applications.hme.abstract_model import AbstractModel


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
            "a": ["a1", "a2", "a1", "a2"],
            "b": ["b1", "b2", "b3", "b1"],
            "c": ["c1", "c2", "c3", "c4"],
            "d": ["d1", "d1", "d1", "d1"],
            "e": ["e1", "e1", "e2", "e3"],
        }
    )


@pytest.mark.parametrize(
    "formula, fixed_effects, random_effects",
    [
        (
            "y ~ 1 + x + (1|a) + (1|b/c) + (1|d+e)",
            ["1", "x"],
            ["a", "b", ("b", "c"), ("d", "e")],
        ),
    ],
)
def test_parse_formula(formula, fixed_effects, random_effects):
    feff, reff = AbstractModel.parse_formula(formula=formula)
    assert feff == fixed_effects
    assert reff == random_effects


@pytest.mark.parametrize(
    "formula, fixed_effects, random_effects",
    [
        # random effects
        (
            "y ~ 1 + x + (1|a) + (1|b/c) + (1|d:e)",
            ["Intercept", "x"],
            ["a", "b", ("b", "c"), ("d", "e")],
        ),
        # categorical fixed effects
        (
            "y ~ 1 + a + b + c",
            [
                "Intercept",
                "a[T.a2]",
                "b[T.b2]",
                "b[T.b3]",
                "c[T.c2]",
                "c[T.c3]",
                "c[T.c4]",
            ],
            [],
        ),
        # no intercept and missing explicit outcome
        (
            "~ 0 + a + b + (1|c/d)",
            ["a[a1]", "a[a2]", "b[T.b2]", "b[T.b3]"],
            ["c", ("c", "d")],
        ),
        # transformation on predictor variables
        (
            "y ~ 1 + center(x) + a + (1|b/c)",
            ["Intercept", "a[T.a2]", "center(x)"],
            ["b", ("b", "c")],
        ),
    ],
)
def test_parse_formula_patsy(data, formula, fixed_effects, random_effects):
    model = HME(
        data=data,
        model_config=ModelConfig(mean_regression=RegressionConfig(formula=formula)),
    )
    assert model.model.fixed_effects == fixed_effects
    assert model.model.random_effects == random_effects


@pytest.mark.parametrize(
    "formula, outcome",
    [
        (
            "np.sqrt(y) ~ x + (1|a)",
            "y",
        )
    ],
)
def test_outcome_inconsistency_exception(data, formula, outcome):
    mean_config = RegressionConfig(outcome=outcome, formula=formula)
    with pytest.raises(ValueError, match="Inconsistent outcome variable"):
        HME(
            data=data,
            model_config=ModelConfig(mean_regression=mean_config),
        )
