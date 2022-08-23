# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Tuple, TypeVar

from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)

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
    except Exception as e:
        logger.warn(
            f"Fails to initialize NNC due to the following error: {str(e)}\n"
            "Falling back to default inference engine."
        )
        # return original function without change
        return f

    return raw_nnc_jit(f, static_argnums)


__all__ = ["nnc_jit"]
