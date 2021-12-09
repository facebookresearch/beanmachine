# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist
from beanmachine.ppl.world.variable import Variable


def test_log_prob():
    var1 = Variable(value=torch.zeros(3), distribution=dist.Bernoulli(0.8))
    # verify that the cached property `log_prob` is recomputed when we replace the
    # fields of a Variable
    var2 = var1.replace(value=torch.ones(3))
    assert var1.log_prob.sum() < var2.log_prob.sum()

    var3 = var1.replace(distribution=dist.Normal(0.0, 1.0))
    assert var1.log_prob.sum() < var3.log_prob.sum()

    var4 = var1.replace(distribution=dist.Categorical(logits=torch.rand(2, 4)))
    assert torch.all(torch.isinf(var4.log_prob))

    var5 = Variable(
        value=torch.tensor(10).double(),
        distribution=dist.Uniform(
            torch.tensor(0.0).double(), torch.tensor(1.0).double()
        ),
    )
    # Check that the log prob has the right dtype
    assert var5.log_prob.dtype == torch.double
    assert torch.isinf(var5.log_prob)

    var6 = var5.replace(value=torch.tensor(1))
    assert torch.isinf(var6.log_prob)
