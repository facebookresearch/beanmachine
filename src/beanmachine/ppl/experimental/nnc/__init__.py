# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import Callable, Optional, Tuple, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def nnc_jit(
    f: Callable[P, R], static_argnums: Optional[Tuple[int]] = None
) -> Callable[P, R]:
    """
    A helper function that lazily imports the NNC utils, which initialize the compiler
    and displaying a experimental warning, then invoke the underlying nnc_jit on
    the function f.
    """
    try:
        # The setup code in `nnc.utils` will only be executed once in a Python session
        from beanmachine.ppl.experimental.nnc.utils import nnc_jit as raw_nnc_jit
    except ImportError as e:
        if sys.platform.startswith("win"):
            message = "functorch is not available on Windows."
        else:
            message = (
                "Fails to initialize NNC. This is likely caused by version mismatch "
                "between PyTorch and functorch. Please checkout the functorch project "
                "for installation guide (https://github.com/pytorch/functorch)."
            )
        raise RuntimeError(message) from e

    return raw_nnc_jit(f, static_argnums)


__all__ = ["nnc_jit"]
