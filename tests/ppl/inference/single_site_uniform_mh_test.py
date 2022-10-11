# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.examples.conjugate_models import (
    BetaBernoulliModel,
    CategoricalDirichletModel,
)
from torch import tensor

from ..utils.fixtures import parametrize_inference, parametrize_model


pytestmark = parametrize_inference([bm.SingleSiteUniformMetropolisHastings()])


@parametrize_model(
    [
        BetaBernoulliModel(tensor(2.0), tensor(2.0)),
        CategoricalDirichletModel(tensor([0.5, 0.5])),
    ]
)
def test_single_site_uniform_mh(model, inference):
    p_key = model.theta()
    l_key = model.x(0) if model.x_dim == 1 else model.x()
    sampler = inference.sampler([p_key], {l_key: torch.tensor(0.0)}, num_samples=5)
    for world in sampler:
        assert p_key in world
        assert l_key in world
        assert p_key in world.get_variable(l_key).parents
        assert l_key in world.get_variable(p_key).children
