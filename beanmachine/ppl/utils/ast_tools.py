# Copyright (c) Facebook, Inc. and its affiliates.
"""Debugging tools for working with ASTs"""

import ast
from ast import AST
from typing import Any, List, Tuple

import beanmachine.ppl.utils.dotbuilder as db
import beanmachine.ppl.utils.treeprinter as tp


def _get_name(node: Any) -> str:
    if isinstance(node, list):
        return "list"
    if isinstance(node, AST):
        return type(node).__name__
    return str(node)


def print_tree(node: AST, unicode: bool = True) -> str:
    def get_children(node: Any) -> List[Any]:
        if isinstance(node, list):
            return node
        if isinstance(node, AST):
            return [child for (name, child) in ast.iter_fields(node)]
        return []

    return tp.print_tree(node, get_children, _get_name, unicode)


def print_graph(node: AST) -> str:
    def get_children(node: Any) -> List[Tuple[str, Any]]:
        if isinstance(node, list):
            return [(str(i), a) for i, a in enumerate(node)]
        if isinstance(node, AST):
            return list(ast.iter_fields(node))
        return []

    return db.print_graph([node], get_children, None, _get_name)
