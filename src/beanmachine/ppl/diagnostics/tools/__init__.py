# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa

"""Visual diagnostic tools for Bean Machine models."""

import sys
from pathlib import Path


if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


TOOLS_DIR = Path(__file__).parent.resolve()
JS_DIR = TOOLS_DIR.joinpath("js")
JS_DIST_DIR = JS_DIR.joinpath("dist")
