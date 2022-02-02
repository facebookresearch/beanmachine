# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, List

import beanmachine.ppl.compiler.bmg_nodes as bn
import torch.distributions as dist
from beanmachine.ppl.compiler.bmg_nodes import BMGNode


def only_ordinary_arguments(args, kwargs) -> bool:
    if any(isinstance(arg, BMGNode) for arg in args):
        return False
    if any(isinstance(arg, BMGNode) for arg in kwargs.values()):
        return False
    return True


def _get_ordinary_value(x: Any) -> Any:
    return x.value if isinstance(x, bn.ConstantNode) else x


def _is_standard_normal(x: Any) -> bool:
    return isinstance(x, dist.Normal) and x.mean == 0.0 and x.stddev == 1.0


def _is_phi(f: Any, arguments: List[Any], kwargs: Dict[str, Any]) -> bool:
    # We need to know if this call is Normal.cdf(Normal(0.0, 1.0), x).
    # (Note that we have already rewritten Normal(0.0, 1.0).cdf(x) into
    # this form.)
    # TODO: Support kwargs
    if f is not dist.Normal.cdf or len(arguments) < 2:
        return False
    s = arguments[0]
    return isinstance(s, dist.Normal) and s.mean == 0.0 and s.stddev == 1.0


def _flatten_all_lists(xs):
    """Takes a list-of-lists, with arbitrary nesting level;
    returns an iteration of all elements."""
    if isinstance(xs, list):
        for x in xs:
            yield from _flatten_all_lists(x)
    else:
        yield xs


def _list_to_zeros(xs):
    """Takes a list-of-lists, with arbitrary nesting level;
    returns a list-of-lists of the same shape but with every non-list
    element replaced with zero."""
    if isinstance(xs, list):
        return [_list_to_zeros(x) for x in xs]
    return 0


def _hashable(x: Any) -> bool:
    # Oddly enough, Python does not allow you to test for set inclusion
    # if the object is not hashable. Since it is impossible for an unhashable
    # object to be in a set, Python could simply say no when asked if a set
    # contains any unhashable object. It does not, so we are forced to do so.

    # All hashable objects have a callable __hash__ attribute.
    if not hasattr(x, "__hash__"):
        return False
    if not isinstance(x.__hash__, Callable):
        return False

    # It is possible that callable __hash__ exists but throws, which makes it
    # unhashable. Eliminate that possibility as well.
    try:
        hash(x)
    except Exception:
        return False

    return True


known_tensor_instance_functions = [
    "add",
    "div",
    "exp",
    "expm1",
    "float",
    "int",
    "item",
    "log",
    "logical_not",
    "logsumexp",
    "matmul",
    "mm",
    "mul",
    "neg",
    "pow",
    "sigmoid",
    "sub",
]


# This helper class is to solve a problem in the simulated
# execution of the model during graph accumulation. Consider
# a model fragment like:
#
#   n = normal()
#   y = n.exp()
#
# During graph construction, n will be a SampleNode whose
# operand is a NormalNode, but SampleNode does not have a
# method "exp".
#
# The lifted program we execute will be something like:
#
#   n = bmg.handle_function(normal, [])
#   func = bmg.handle_dot(n, "exp")
#   y = bmg.handle_function(func, [])
#
# The "func" that is returned is one of these KnownFunction
# objects, which captures the notion "I am an invocation
# of known function Tensor.exp on a receiver that is a BMG
# node".  We then turn that into a exp node in handle_function.


class KnownFunction:
    receiver: BMGNode
    function: Callable

    def __init__(self, receiver: BMGNode, function: Callable) -> None:
        if not isinstance(receiver, BMGNode):
            raise TypeError(
                f"KnownFunction receiver must be BMGNode but is {type(receiver)}"
            )
        if not isinstance(function, Callable):
            raise TypeError(
                f"KnownFunction function must be Callable but is {type(function)}"
            )

        self.receiver = receiver
        self.function = function
