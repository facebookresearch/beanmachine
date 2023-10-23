# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
from sts.util import optional


def test_optional():
    entered = False
    with optional(False, open, "dummy", "r") as f:
        assert f is None
        entered = True
    assert entered
    entered = False
    with pytest.raises(FileNotFoundError):
        with optional(True, open, "dummy", "r") as f:
            pass
