# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch
import torch.distributions as dist


@bm.random_variable
def foo():
    return dist.Normal(0.0, 1.0)


def test_set_random_seed():
    def sample_with_seed(seed):
        bm.seed(seed)
        return bm.SingleSiteAncestralMetropolisHastings().infer(
            [foo()], {}, num_samples=20, num_chains=1
        )

    samples1 = sample_with_seed(123)
    samples2 = sample_with_seed(123)
    assert torch.allclose(samples1[foo()], samples2[foo()])
