# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.jit
from functorch.compile import nnc_jit


# the warning will only be shown to user once when this module is imported
warnings.warn(
    "The support of NNC compiler is experimental and the API is subject to"
    "change in the future releases of Bean Machine. For questions regarding NNC, please"
    "checkout the functorch project (https://github.com/pytorch/functorch)."
)

# allows reductions to be compiled by NNC
# pyre-fixme[16]: Module `_C` has no attribute `_jit_set_texpr_reductions_enabled`.
torch._C._jit_set_texpr_reductions_enabled(True)


__all__ = ["nnc_jit"]
