# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
from beanmachine.ppl.examples.primitive_models import NormalModel
from torch import tensor

from ..utils.fixtures import approx_all, parametrize_inference, parametrize_model


pytestmark = [
    parametrize_model([NormalModel(tensor(0.0), tensor(1.0))]),
    parametrize_inference([bm.SingleSiteAncestralMetropolisHastings()]),
]


@pytest.mark.parametrize("seed", [123, 47])
def test_set_random_seed(model, inference, seed):
    def sample_with_seed(seed):
        bm.seed(seed)
        return inference.infer([model.x()], {}, num_samples=20, num_chains=1)

    samples = (sample_with_seed(seed) for _ in range(2))
    assert approx_all(*(s[model.x()] for s in samples))


def test_detach_samples(model, inference):
    """Test to ensure samples are detached from torch computation graphs."""
    queries = [model.x()]
    samples = inference.infer(
        queries=queries,
        observations={},
        num_samples=20,
        num_chains=1,
    )
    rv_data = samples[model.x()]
    idata = samples.to_inference_data()
    assert hasattr(rv_data, "detach")
    assert not hasattr(idata["posterior"][model.x()], "detach")
