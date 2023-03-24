# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.examples.primitive_models import GammaModel
from beanmachine.ppl.inference.proposer.nmc import SingleSiteHalfSpaceNMCProposer
from beanmachine.ppl.world import World
from torch import tensor

from ....utils.fixtures import approx, parametrize_model, parametrize_proposer


@parametrize_model([GammaModel(tensor([2.0, 2.0, 2.0]), tensor([2.0, 2.0, 2.0]))])
@parametrize_proposer([SingleSiteHalfSpaceNMCProposer])
def test_alpha_and_beta_for_gamma(model, proposer):
    world = World()
    with world:
        model.x()
    prop = proposer(model.x())
    is_valid, predicted_alpha, predicted_beta = prop.compute_alpha_beta(world)
    assert is_valid
    assert approx(model.alpha.sum(), predicted_alpha.sum(), 1e-4)
    assert approx(model.beta.sum(), predicted_beta.sum(), 1e-4)
