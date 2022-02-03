# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functorch
import torch
import torch.jit
import torch.utils._pytree as pytree
from functorch.compile import nop, aot_function

# override the usage of torch.jit.script, which has a bit of issue handling
# empty lists (functorch#440)
def simple_ts_compile(fx_g, example_inps):
    f = torch.jit.trace(fx_g, example_inps, strict=False)
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f, static_argnums=None):
    return aot_function(f, simple_ts_compile, nop, static_argnums=static_argnums)


functorch._src.compilers.simple_ts_compile = simple_ts_compile


# override default dict flatten (which requires keys to be sortable)
def _dict_flatten(d):
    keys = list(d.keys())
    values = [d[key] for key in keys]
    return values, keys


def _dict_unflatten(values, context):
    return {key: value for key, value in zip(context, values)}


pytree._register_pytree_node(dict, _dict_flatten, _dict_unflatten)

__all__ = ["nnc_jit"]
