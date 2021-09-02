# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import warnings

import pytest
from beanmachine.applications.hme import ModelConfig
from beanmachine.applications.hme.abstract_model import AbstractModel


class RealizedModel(AbstractModel):
    def build_graph(self):
        return super().build_graph()


@pytest.mark.parametrize(
    "post_samples",
    [
        [
            [[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]],
            [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
        ],
    ],
)
def test_get_bmg_diagnostics(post_samples):
    model = RealizedModel(data=None, model_config=ModelConfig())

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        post_diagnostics = model._get_bmg_diagnostics(post_samples)

        assert len(w) == 1
        assert "NaN encountered" in str(w[-1].message)
        assert "N_Eff" not in post_diagnostics.columns
