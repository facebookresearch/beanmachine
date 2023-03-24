# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.examples.conjugate_models import NormalNormalModel
from torch import tensor

from ..utils.fixtures import parametrize_inference, parametrize_model


pytestmark = parametrize_model(
    [NormalNormalModel(tensor(0.0), tensor(1.0), tensor(1.0))]
)


@parametrize_inference(
    [
        bm.GlobalNoUTurnSampler(nnc_compile=True),
        bm.GlobalHamiltonianMonteCarlo(trajectory_length=1.0, nnc_compile=True),
    ]
)
def test_nnc_compile(model, inference):
    queries = [model.theta()]
    observations = {model.x(): torch.tensor(0.5)}
    num_samples = 30
    num_chains = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # verify that NNC can run through
        samples = inference.infer(
            queries,
            observations,
            num_samples,
            num_adaptive_samples=num_samples,
            num_chains=num_chains,
        )
    # sanity check: make sure that the samples are valid
    assert not torch.isnan(samples[model.theta()]).any()
