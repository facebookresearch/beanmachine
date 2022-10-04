# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import gpytorch
from beanmachine.ppl.world import get_world_context


def _trace_bm(module, name_to_rv=None, is_tracing=True, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if name_to_rv is None:
        name_to_rv = {}
    if hasattr(module, "_priors"):
        for prior_name, (prior, closure, setting_closure) in module._priors.items():
            if prior is not None and prior not in memo:
                if setting_closure is None:
                    raise RuntimeError(
                        "Cannot perform fully Bayesian inference without a setting_closure for each prior,"
                        f" but the following prior had none: {prior_name}, {prior}."
                    )
                memo.add(prior_name)
                prior = prior.expand(closure(module).shape)
                rv_name = prefix + ("." if prefix else "") + prior_name
                if is_tracing:
                    # tracing pass, no enclosing World
                    def f():
                        return prior

                    f.__name__ = rv_name
                    rv = bm.random_variable(f)
                    name_to_rv[rv_name] = rv()
                else:
                    # sampling pass, must be enclosed by World
                    world = get_world_context()
                    assert (
                        world is not None
                    ), "Expected enclosing World context for bm.random_variable priors"
                    value = world.update_graph(name_to_rv[rv_name])
                    setting_closure(module, value)

    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        _, child_name_to_rv = _trace_bm(
            module=module_,
            name_to_rv=name_to_rv,
            is_tracing=is_tracing,
            memo=memo,
            prefix=submodule_prefix,
        )
        name_to_rv.update(child_name_to_rv)

    return module, name_to_rv


def make_prior_random_variables(
    module,
    name_to_rv=None,
    memo=None,
    prefix="",
):
    """
    Recurses through `module` and its childrens' `._priors`, creating `bm.random_variable`s
    for each prior. Returns a map from prior names to `random_variable`s.
    """
    return _trace_bm(module, name_to_rv=None, is_tracing=True)[1]


def bm_sample_from_prior(model, name_to_rv) -> gpytorch.module.Module:
    """
    Samples from `model` with parameters drawn by invoking the
    `random_variable` to their prior in `name_to_rv`.
    """
    return _trace_bm(model, name_to_rv, is_tracing=False)[0]
