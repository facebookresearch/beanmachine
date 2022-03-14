# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functorch
import torch
import torch.jit
import torch.utils._pytree as pytree
from functorch.compile import nop, aot_function, decomposition_table, register_decomposition

torch._C._jit_set_texpr_reductions_enabled(True)

# override the usage of torch.jit.script, which has a bit of issue handling
# empty lists (functorch#440)
def simple_ts_compile(fx_g, example_inps):
    f = torch.jit.trace(fx_g, example_inps, strict=False)
    f = torch.jit.freeze(f.eval())
    torch._C._jit_pass_remove_mutation(f.graph)

    for _ in range(5):
        f(*example_inps)
    print(f.graph_for(*example_inps))
    return f


aten = torch.ops.aten
decompositions = [aten.detach.default]
bm_decompositions = {k: v for k,v in decomposition_table.items() if k in decompositions}

@register_decomposition(aten.mv, bm_decompositions)
def mv(a, b):
    return (a * b).sum(dim=-1)

@register_decomposition(aten.dot, bm_decompositions)
def dot(a, b):
    return (a * b).sum(dim=-1)

@register_decomposition(aten.nan_to_num, bm_decompositions)
def nan_to_num(a, val):
    return aten.where(a != a, val, a)

@register_decomposition(aten.zeros_like, bm_decompositions)
def zeros_like(a, **kwargs):
    return a * 0

@register_decomposition(aten.ones_like, bm_decompositions)
def ones_like(a, **kwargs):
    return a * 0 + 1

def nnc_jit(f, static_argnums=None):
    return aot_function(f, simple_ts_compile, nop, static_argnums=static_argnums, decompositions=bm_decompositions)

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
