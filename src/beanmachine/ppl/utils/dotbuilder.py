#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A builder for the graphviz DOT language"""
import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from beanmachine.ppl.utils.treeprinter import _is_named_tuple, _to_string
from beanmachine.ppl.utils.unique_name import make_namer


def _get_children(n: Any) -> List[Tuple[str, Any]]:
    if isinstance(n, dict):
        return list(n.items())
    if _is_named_tuple(n):
        return [(k, getattr(n, k)) for k in type(n)._fields]
    if isinstance(n, tuple) or isinstance(n, list):
        return [(str(ind), item) for (ind, item) in enumerate(n)]
    return []


def print_graph(
    roots: List[Any],
    get_children: Callable[[Any], List[Tuple[str, Any]]] = _get_children,
    to_node_name: Optional[Callable[[Any], str]] = None,
    to_label: Callable[[Any], str] = _to_string,
) -> str:
    """
    This function converts an object representing a graph into a string
    in the DOT graph display language.

    The roots are a set of nodes in the graph; the final graph description will
    contain the transitive closure of the children of all roots.

    get_children returns a list of (edge_label, node) pairs; if no argument
    is supplied then a default function that can handle lists, tuples and
    dictionaries is used.

    to_node_name returns a *unique* string used to identify the node in the
    graph.

    to_label gives a *not necessarily unique* label for a node in a graph.
    Again if not supplied, a default that can handle dictionaries, lists and
    tuples is used.
    """

    tnn = make_namer(to_node_name, "N")

    builder: DotBuilder = DotBuilder()
    stack: List[Any] = []
    stack.extend(roots)
    done: Set[str] = set()
    for root in roots:
        builder.with_node(tnn(root), to_label(root))
    while len(stack) > 0:
        current = stack.pop()
        current_node = tnn(current)
        if current_node not in done:
            for (edge_label, child) in get_children(current):
                child_node = tnn(child)
                builder.with_node(child_node, to_label(child))
                builder.with_edge(current_node, child_node, edge_label)
                stack.append(child)
            done.add(current_node)
    return str(builder)


class DotBuilder:

    name: str
    is_subgraph: bool
    is_cluster: bool
    _label: str
    _node_map: "Dict[str, DotNode]"
    _edges: "Set[DotEdge]"
    _comments: List[str]
    _subgraphs: "List[DotBuilder]"
    _nodes: "List[DotNode]"
    _current_subgraph: "Optional[DotBuilder]"

    def __init__(
        self, name: str = "graph", is_subgraph: bool = False, is_cluster: bool = False
    ):
        self.name = name
        self.is_subgraph = is_subgraph
        self.is_cluster = is_cluster
        self._label = ""
        self._node_map = {}
        self._edges = set()
        self._comments = []
        self._subgraphs = []
        self._nodes = []
        self._current_subgraph = None

    def with_label(self, label: str) -> "DotBuilder":
        sg = self._current_subgraph
        if sg is None:
            self._label = label
        else:
            sg.with_label(label)
        return self

    def start_subgraph(self, name: str, is_cluster: bool) -> "DotBuilder":
        sg = self._current_subgraph
        if sg is None:
            csg = DotBuilder(name, True, is_cluster)
            self._current_subgraph = csg
            self._subgraphs.append(csg)
        else:
            sg.start_subgraph(name, is_cluster)
        return self

    def end_subgraph(self) -> "DotBuilder":
        sg = self._current_subgraph
        if sg is None:
            raise ValueError("Cannot end a non-existing subgraph.")
        elif sg._current_subgraph is None:
            self._current_subgraph = None
        else:
            sg.end_subgraph()
        return self

    def _get_node(self, name: str) -> "DotNode":
        if name in self._node_map:
            return self._node_map[name]
        new_node = DotNode(name, "", "")
        self._node_map[name] = new_node
        self._nodes.append(new_node)
        return new_node

    def with_comment(self, comment: str) -> "DotBuilder":
        sg = self._current_subgraph
        if sg is None:
            self._comments.append(comment)
        else:
            sg.with_comment(comment)
        return self

    def with_node(self, name: str, label: str, color: str = "") -> "DotBuilder":
        sg = self._current_subgraph
        if sg is None:
            n = self._get_node(name)
            n.label = label
            n.color = color
        else:
            sg.with_node(name, label, color)
        return self

    def with_edge(
        self,
        frm: str,
        to: str,
        label: str = "",
        color: str = "",
        constrained: bool = True,
    ) -> "DotBuilder":
        sg = self._current_subgraph
        if sg is None:
            f = self._get_node(frm)
            t = self._get_node(to)
            self._edges.add(DotEdge(f, t, label, color, constrained))
        else:
            sg.with_edge(frm, to, label, color, constrained)
        return self

    def _to_string(self, indent: str, sb: List[str]) -> List[str]:
        new_indent = indent + "  "
        sb.append(indent)
        sb.append("subgraph" if self.is_subgraph else "digraph")
        i = ""
        has_name = len(self.name) > 0
        if has_name and self.is_cluster:
            i = smart_quote("cluster_" + self.name)
        elif has_name:
            i = smart_quote(self.name)
        elif self.is_cluster:
            i = "cluster"
        if len(i) > 0:
            sb.append(" " + i)
        sb.append(" {\n")
        for c in self._comments:
            sb.append(new_indent + "// " + c + "\n")
        if len(self._label) > 0:
            sb.append(new_indent + "label=" + smart_quote(self._label) + "\n")
        nodes = sorted(new_indent + str(n) + "\n" for n in self._nodes)
        sb.extend(nodes)
        edges = sorted(new_indent + str(e) + "\n" for e in self._edges)
        sb.extend(edges)
        for db in self._subgraphs:
            sb = db._to_string(new_indent, sb)
        sb.append(indent + "}\n")
        return sb

    def __str__(self):
        return "".join(self._to_string("", []))


class DotNode:
    name: str
    label: str
    color: str

    def __init__(self, name: str, label: str, color: str):
        self.name = name
        self.label = label
        self.color = color

    def __str__(self) -> str:
        props: List[str] = []
        if len(self.label) != 0 and self.label != self.name:
            props.append("label=" + smart_quote(self.label))
        if len(self.color) != 0:
            props.append("color=" + smart_quote(self.label))
        p = "" if len(props) == 0 else "[" + " ".join(props) + "]"
        return smart_quote(self.name) + p + ";"


class DotEdge:
    frm: DotNode
    to: DotNode
    label: str
    color: str
    constrained: bool

    def __init__(
        self, frm: DotNode, to: DotNode, label: str, color: str, constrained: bool
    ):
        self.frm = frm
        self.to = to
        self.label = label
        self.color = color
        self.constrained = constrained

    def __str__(self) -> str:
        props: List[str] = []
        if len(self.label) != 0:
            props.append("label=" + smart_quote(self.label))
        if len(self.color) != 0:
            props.append("color=" + smart_quote(self.label))
        if not self.constrained:
            props.append("constraint=false")
        p = "" if len(props) == 0 else "[" + " ".join(props) + "]"
        return smart_quote(self.frm.name) + " -> " + smart_quote(self.to.name) + p + ";"


_keywords: List[str] = ["digraph", "edge", "graph", "node", "strict", "subgraph"]
_alphanum = re.compile("^[A-Za-z_][A-Za-z_0-9]*$")
_numeric = re.compile("^[-]?(\\.[0-9]+|[0-9]+(\\.[0-9]*)?)$")


def smart_quote(s: str) -> str:
    if s is None or len(s) == 0:
        return '""'
    if s.lower() in _keywords:
        return json.dumps(s)
    if _alphanum.match(s):
        return s
    if _numeric.match(s):
        return s
    return json.dumps(s)
