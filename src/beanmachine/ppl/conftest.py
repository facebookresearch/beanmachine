# Copyright (c) Facebook, Inc. and its affiliates.
import beanmachine.ppl as bm
import pytest


@pytest.fixture(autouse=True)
def fix_random_seed():
    """Fix the random state for every test in the test suite"""
    bm.seed(0)
