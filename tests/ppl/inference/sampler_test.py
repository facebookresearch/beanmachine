# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.examples.conjugate_models import NormalNormalModel
from torch import tensor

from ..utils.fixtures import (
    parametrize_inference,
    parametrize_inference_comparison,
    parametrize_model,
)


pytestmark = parametrize_model(
    [NormalNormalModel(tensor(0.0), tensor(1.0), tensor(1.0))]
)


@parametrize_inference([bm.GlobalNoUTurnSampler(), bm.GlobalHamiltonianMonteCarlo(1.0)])
def test_sampler(model, inference):
    queries = [model.theta()]
    observations = {model.x(): torch.tensor(0.5)}
    num_samples = 10
    sampler = inference.sampler(
        queries, observations, num_samples, num_adaptive_samples=0
    )
    worlds = list(sampler)
    assert len(worlds) == num_samples
    for world in worlds:
        assert model.theta() in world
        with world:
            assert isinstance(model.theta(), torch.Tensor)


@parametrize_inference_comparison(
    [bm.GlobalNoUTurnSampler(), bm.GlobalHamiltonianMonteCarlo(1.0)]
)
def test_two_samplers(model, inferences):
    queries = [model.theta()]
    observations = {model.x(): torch.tensor(0.5)}
    samplers = [alg.sampler(queries, observations) for alg in inferences]
    world = next(samplers[0])
    # it's possible to use multiple sampler interchangably to update the worlds (or
    # in general, pass a new world to sampler and continue inference with existing
    # hyperparameters)
    for _ in range(3):
        world = samplers[1].send(world)
        world = samplers[0].send(world)
    assert model.theta() in world
    assert model.x() in world
