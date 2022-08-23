# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.vi import VariationalInfer

cpu_device = torch.device("cpu")


class NormalNormal:
    def __init__(self, device: Optional[torch.device] = cpu_device):
        self.device = device

    @bm.random_variable
    def mu(self):
        return dist.Normal(
            torch.zeros(1).to(self.device), 10 * torch.ones(1).to(self.device)
        )

    @bm.random_variable
    def x(self, i):
        return dist.Normal(self.mu(), torch.ones(1).to(self.device))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU access to train the model"
)
def test_normal_normal_guide_step_gpu():
    device = torch.device("cuda:0")
    model = NormalNormal(device=device)

    @bm.param
    def phi():
        return torch.zeros(2).to(device)  # mean, log std

    @bm.random_variable
    def q_mu():
        params = phi()
        return dist.Normal(params[0], params[1].exp())

    world = VariationalInfer(
        queries_to_guides={model.mu(): q_mu()},
        observations={
            model.x(1): torch.tensor(9.0),
            model.x(2): torch.tensor(10.0),
        },
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-1),
        device=device,
    ).infer(
        num_steps=1000,
    )
    mu_approx = world.get_variable(q_mu()).distribution

    assert (mu_approx.mean - 9.6).norm() < 1.0
    assert (mu_approx.stddev - 0.7).norm() < 0.3
