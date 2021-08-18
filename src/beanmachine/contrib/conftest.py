# Copyright (c) Facebook, Inc. and its affiliates.
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def random_seed():
    """Fix the random state for every test in the test suite"""
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
