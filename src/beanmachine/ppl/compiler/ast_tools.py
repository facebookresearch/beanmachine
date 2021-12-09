# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Debugging tools for working with ASTs"""

# This module just has some helpful tools that can be used for visualizing
# Python ASTs when debugging the compilation process.

import ast
from ast import AST
from typing import Any, List, Tuple

import beanmachine.ppl.utils.dotbuilder as db
import beanmachine.ppl.utils.treeprinter as tp
import black


def _get_name(node: Any) -> str:
    if isinstance(node, list):
        return "list"
    if isinstance(node, AST):
        return type(node).__name__
    return str(node)


def print_tree(node: AST, unicode: bool = True) -> str:
    """Takes an AST and produces a string containing a hierarchical
    view of the tree structure."""

    def get_children(node: Any) -> List[Any]:
        if isinstance(node, list):
            return node
        if isinstance(node, AST):
            return [child for (name, child) in ast.iter_fields(node)]
        return []

    return tp.print_tree(node, get_children, _get_name, unicode)


def print_graph(node: AST) -> str:
    """Takes an AST and produces a string containing a DOT
    representation of the tree as a graph."""

    def get_children(node: Any) -> List[Tuple[str, Any]]:
        if isinstance(node, list):
            return [(str(i), a) for i, a in enumerate(node)]
        if isinstance(node, AST):
            return list(ast.iter_fields(node))
        return []

    return db.print_graph([node], get_children, None, _get_name)


def print_python(node: AST) -> str:
    """Takes an AST and produces a string containing a human-readable
    Python expression that builds the AST node."""
    return black.format_str(ast.dump(node), mode=black.FileMode())
