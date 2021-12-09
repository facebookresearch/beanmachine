# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa
import sys
from pathlib import Path

tutorials_dir = Path.cwd().parent
sys.path.insert(0, str(tutorials_dir))


import etl
import plots
