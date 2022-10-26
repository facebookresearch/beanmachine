# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from enum import Enum

from typing import Callable

from beanmachine.ppl.inference.proposer.nnc import nnc_jit


class TorchJITBackend(Enum):
    NONE = "none"
    NNC = "nnc"
    INDUCTOR = "inductor"


# TODO (T135789755): update the API to select between backends when we move this
# integration out of experimental.
def get_backend(
    nnc_compile: bool, experimental_inductor_compile: bool
) -> TorchJITBackend:
    """A helper function to select between the Torch JIT backends based on the
    flags"""
    if experimental_inductor_compile:
        if nnc_compile:
            warnings.warn(
                "Overriding nnc_compile option with experimental_inductor_compile",
                stacklevel=3,
            )
        warnings.warn(
            "The support of TorchInductor is experimental and the API is "
            "subject to change in the future releases of Bean Machine. For "
            "questions regarding TorchInductor, please see "
            "https://github.com/pytorch/torchdynamo.",
            stacklevel=3,
        )
        return TorchJITBackend.INDUCTOR
    elif nnc_compile:
        return TorchJITBackend.NNC
    else:
        return TorchJITBackend.NONE


def inductor_jit(f: Callable) -> Callable:
    """
    A helper function that lazily imports the TorchInductor utils and the
    related libraries, then invoke functorch to JIT compile the provided
    function.
    """
    # Lazily import related libraries so users don't have them (e.g. from using
    # an older version of PyTorch) won't run into ModuleNotFound error when
    # importing Bean Machine
    from functorch.compile import aot_function
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.decomposition import select_decomp_table

    return aot_function(f, compile_fx_inner, decompositions=select_decomp_table())


def jit_compile(f: Callable, backend: TorchJITBackend) -> Callable:
    if backend is TorchJITBackend.NNC:
        return nnc_jit(f)
    elif backend is TorchJITBackend.INDUCTOR:
        return inductor_jit(f)
    else:
        # Fall back to use PyTorch
        return f
