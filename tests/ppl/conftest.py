# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
import torch.distributions as dist


@pytest.fixture(autouse=True)
def fix_random_seed():
    """Fix the random state for every test in the test suite."""
    bm.seed(0)


@pytest.fixture(autouse=True)
def disable_torch_distribution_validation():
    """Disables validation of Torch distribution arguments."""
    dist.Distribution.set_default_validate_args(False)
