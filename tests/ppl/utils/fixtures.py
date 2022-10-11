# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from operator import attrgetter
from typing import Callable, List, Tuple, Type

import pytest
import torch.distributions as dist
from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import BaseProposer
from torch import allclose, is_tensor, isclose, Tensor, tensor


# numerical predicates =========================================================


def _approx(comp_fn: Callable, result: Tensor, expected, atol=1e-8):
    return comp_fn(
        result,
        expected if is_tensor(expected) else tensor(expected).to(result),
        atol=atol,
    )


approx = partial(_approx, isclose)
approx_all = partial(_approx, allclose)


# fixture printers =============================================================


def _is_model(arg):
    return arg.__class__.__name__.endswith("Model")


_is_value_tensor = is_tensor


def _is_value_variable(arg):
    return (
        isinstance(arg, tuple)
        and isinstance(arg[0], dist.Distribution)
        and is_tensor(arg[1])
    )


def _is_value(arg):
    return _is_value_tensor(arg) or _is_value_variable(arg)


def _is_inference(arg):
    return isinstance(arg, BaseInference)


def _is_inferences(args):
    return all(map(_is_inference, args))


def _is_proposer(arg):
    return issubclass(arg, BaseProposer)


_id_empty = ""
_id_model = attrgetter("__class__")


def _id_value_tensor(arg):
    return f"Tensor{tuple(arg.shape)}"


def _id_value_variable(arg):
    return f"Variable{tuple(arg[1].shape)}"


def _id_value(arg):
    return _id_value_tensor(arg) if _is_value_tensor(arg) else _id_value_variable(arg)


_id_inference = attrgetter("__class__")


def _id_inferences(args):
    return f"({','.join([a.__class__.__name__ for a in args])})"


_id_proposer = None  # default printer


def _id_model_value(arg):
    return _id_model(arg) if _is_model(arg) else _id_value(arg)


def _id_value_expected(arg):
    return _id_value(arg) if _is_value(arg) else _id_empty


def _id_model_value_expected(arg):
    return _id_model(arg) if _is_model(arg) else _id_value_expected(arg)


# fixtures =====================================================================


def parametrize_model(models: List):
    assert all(map(_is_model, models))
    return pytest.mark.parametrize("model", models, ids=_id_model)


def parametrize_value(args: List[Tuple]):
    assert all(map(_is_value, args))
    return pytest.mark.parametrize("value", args, ids=_id_value)


def parametrize_model_value(args: List[Tuple]):
    assert all(isinstance(a, tuple) for a in args)
    return pytest.mark.parametrize("model, value", args, ids=_id_model_value)


def parametrize_value_expected(args: List[Tuple]):
    assert all(isinstance(a, tuple) for a in args)
    return pytest.mark.parametrize("value, expected", args, ids=_id_value_expected)


def parametrize_model_value_expected(args: List[Tuple]):
    return pytest.mark.parametrize(
        "model, value, expected", args, ids=_id_model_value_expected
    )


def parametrize_inference(methods: List[BaseInference]):
    assert _is_inferences(methods)
    return pytest.mark.parametrize("inference", methods, ids=_id_inference)


def parametrize_inference_comparison(methods: List[BaseInference]):
    assert _is_inferences(methods)
    return pytest.mark.parametrize("inferences", [methods], ids=_id_inferences)


def parametrize_proposer(methods: List[Type[BaseProposer]]):
    assert all(map(_is_proposer, methods))
    return pytest.mark.parametrize("proposer", methods, ids=_id_proposer)
