# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defines print_tree, a helper function to render Python objects as trees."""
from typing import Any, Callable, List


def _is_named_tuple_type(t: type) -> bool:
    if not isinstance(getattr(t, "_fields", None), tuple):
        return False
    if len(t.__bases__) == 1 and t.__bases__[0] == tuple:
        return True
    return any(_is_named_tuple_type(b) for b in t.__bases__)


def _is_named_tuple(x: Any) -> bool:
    return _is_named_tuple_type(type(x))


def _get_children_key_value(v: Any) -> List[Any]:
    if isinstance(v, dict):
        return list(v.items())
    if isinstance(v, list):
        return v
    if _is_named_tuple(v):
        return [(k, getattr(v, k)) for k in type(v)._fields]
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _get_children(n: Any) -> List[Any]:
    if isinstance(n, dict):
        return list(n.items())
    if isinstance(n, list):
        return n
    if _is_named_tuple(n):
        return [(k, getattr(n, k)) for k in type(n)._fields]
    # for key-value pairs we do not want subtypes of tuple, just tuple.
    if type(n) == tuple and len(n) == 2:
        return _get_children_key_value(n[1])
    if isinstance(n, tuple):
        return list(n)
    return []


def _to_string(n: Any) -> str:
    if isinstance(n, dict):
        return "dict"
    if isinstance(n, list):
        return "list"
    # for key-value pairs we do not want subtypes of tuple, just tuple.
    if type(n) == tuple and len(n) == 2:
        return str(n[0])
    if _is_named_tuple(n):
        return type(n).__name__
    if isinstance(n, tuple):
        return "tuple"
    return str(n)


def print_tree(
    root: Any,
    get_children: Callable[[Any], List[Any]] = _get_children,
    to_string: Callable[[Any], str] = _to_string,
    unicode: bool = True,
) -> str:
    """
    Renders an arbitrary Python object as a tree. This is handy for debugging.

    If you have a specific tree structure imposed on an object, you can pass
    in your own get_children method; if omitted, a function that handles Python
    dictionaries, tuples, named tuples and lists is the default.

    The text of each node is determined by the to_string argument; if omitted
    a default function is used.

    The tree produced uses the Unicode box-drawing characters by default; to
    use straight ASCII characters, pass False for the unicode parameter.
    """

    def pt(node, indent):
        builder.append(to_string(node))
        builder.append("\n")
        children = get_children(node)
        for i in range(len(children)):
            last = i == len(children) - 1
            child = children[i]
            builder.append(indent)
            builder.append(el if last else tee)
            builder.append(dash)
            pt(child, indent + (" " if last else bar) + " ")

    el = "\u2514" if unicode else "+"
    tee = "\u251c" if unicode else "+"
    dash = "\u2500" if unicode else "-"
    bar = "\u2502" if unicode else "|"
    builder = []
    pt(root, "")
    return "".join(builder)
