# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

# Ignore all warnings in this module against using tensor as arguments of random
# variables
pytestmark = pytest.mark.filterwarnings(
    "ignore:PyTorch tensors are hashed by memory address*:UserWarning"
)
