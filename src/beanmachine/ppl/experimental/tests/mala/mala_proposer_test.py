# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.mala.mala_proposer import MALAProposer
from beanmachine.ppl.world import World


@bm.random_variable
def foo():
    return dist.Uniform(0.0, 1.0)


@bm.random_variable
def bar():
    return dist.Normal(foo(), 1.0)


@pytest.fixture
def world():
    w = World()
    w.call(bar())
    return w


@pytest.fixture
def mala(world):
    mala_proposer = MALAProposer(world, world.latent_nodes, 5)
    return mala_proposer


def test_aller_retour(mala):
    """
    Test that the sampler comes back to the initial phase space point after reversing momentum.
    """
    step_size = 0.1
    tolerance = 1e-6
    momentums = mala._initialize_momentums(mala._positions)
    intermediate_positions, intermediate_momentums, pe, pe_grad = mala._leapfrog_step(
        mala._positions, momentums, step_size, mala._mass_inv
    )
    new_positions, new_momentums, pe, pe_grad = mala._leapfrog_step(
        intermediate_positions, intermediate_momentums, -step_size, mala._mass_inv
    )
    for (_init_node, init_z), (_final_node, final_z) in zip(
        mala._positions.items(), new_positions.items()
    ):
        assert torch.isclose(init_z, final_z, atol=tolerance)

    for (_init_node, init_r), (_final_node, final_r) in zip(
        momentums.items(), new_momentums.items()
    ):
        assert torch.isclose(init_r, final_r, atol=tolerance)
