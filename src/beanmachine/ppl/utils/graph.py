#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A mutable graph builder"""
from hashlib import md5
from typing import Callable, Dict, Generic, List, Optional, TypeVar

from beanmachine.ppl.utils.dotbuilder import DotBuilder
from beanmachine.ppl.utils.equivalence import partition_by_kernel
from beanmachine.ppl.utils.unique_name import make_namer


# A plate is a collection of nodes and plates.
# A graph is a single plate with a collection of edges.
# That is, in this model of a graph, only the topmost level
# contains the edges; plates contain no edges.

T = TypeVar("T")  # Node type


class Plate(Generic[T]):
    # Yes, using lists means that we have O(n) removal. But removals are
    # rare, the lists are typically short, and lists guarantee that
    # the enumeration order is deterministic, which means that we get
    # repeatable behavior for testing.

    _plates: "List[Plate[T]]"
    _parent: "Optional[Plate[T]]"
    _graph: "Graph[T]"
    _nodes: List[T]

    def __init__(self, graph: "Graph[T]", parent: "Optional[Plate[T]]") -> None:
        self._plates = []
        self._parent = parent
        self._graph = graph
        self._nodes = []

    def with_plate(self) -> "Plate[T]":
        """Add a new Plate to this Plate; returns the new Plate"""
        sub = Plate(self._graph, self)
        self._plates.append(sub)
        return sub

    def without_plate(self, sub: "Plate[T]") -> "Plate[T]":
        """Removes a given Plate, and all its Plates, and all its nodes."""
        if sub in self._plates:
            # Recursively destroy every nested Plate.
            # We're going to be modifying a collection as we enumerate it
            # so make a copy.
            for subsub in sub._plates.copy():
                sub.without_plate(subsub)
            # Destroy every node.
            for node in sub._nodes.copy():
                sub.without_node(node)
            # Delete the Plate
            self._plates.remove(sub)
        return self

    def with_node(self, node: T) -> "Plate[T]":
        """Adds a new node to the plate, or, if the node is already in the
        graph, moves it to this plate. Edges are unaffected by moves."""
        if node not in self._nodes:
            # Remove the node from its current Plate.
            if node in self._graph._nodes:
                self._graph._nodes[node]._nodes.remove(node)
            # Let the graph know that this node is in this Plate.
            self._graph._nodes[node] = self
            self._nodes.append(node)
            # If this is a new node, set its incoming and outgoing
            # edge sets to empty. If it is not a new node, keep
            # them the same.
            if node not in self._graph._outgoing:
                self._graph._outgoing[node] = []
            if node not in self._graph._incoming:
                self._graph._incoming[node] = []
        return self

    def without_node(self, node: T) -> "Plate[T]":
        if node in self._nodes:
            # Remove the node
            del self._graph._nodes[node]
            self._nodes.remove(node)
            # Delete all the edges associated with this node
            for o in list(self._graph._outgoing[node]):
                self._graph._incoming[o].remove(node)
            for i in list(self._graph._incoming[node]):
                self._graph._outgoing[i].remove(node)
            del self._graph._outgoing[node]
            del self._graph._incoming[node]
        return self

    def with_edge(self, start: T, end: T) -> "Plate[T]":
        if start not in self._graph._nodes:
            self.with_node(start)
        if end not in self._graph._nodes:
            self.with_node(end)
        self._graph._incoming[end].append(start)
        self._graph._outgoing[start].append(end)
        return self


class Graph(Generic[T]):
    _nodes: Dict[T, Plate[T]]
    _outgoing: Dict[T, List[T]]
    _incoming: Dict[T, List[T]]
    _top: Plate[T]
    _to_name: Callable[[T], str]
    _to_label: Callable[[T], str]
    _to_kernel: Callable[[T], str]

    def __init__(
        self,
        to_name: Optional[Callable[[T], str]] = None,
        to_label: Callable[[T], str] = str,
        to_kernel: Callable[[T], str] = str,
    ):
        # to_name gives a *unique* name to a node.
        # to_label gives a *not necessarily unique* label when *displaying* a graph.
        # to_kernel gives a string that is always the same if two nodes are to
        #   be treated as isomorphic. This lets us make labels in the output that
        #   are different than the isomorphism kernel.
        self._nodes = {}
        self._outgoing = {}
        self._incoming = {}
        self._top = Plate(self, None)
        self._to_name = make_namer(to_name, "N")
        self._to_label = to_label
        self._to_kernel = to_kernel

    def with_plate(self) -> "Plate[T]":
        """Add a plate to the top level; returns the plate"""
        return self._top.with_plate()

    def without_plate(self, sub: Plate[T]) -> "Graph[T]":
        """Removes a plate from the top level, and all its plates, and
        all its nodes."""
        self._top.without_plate(sub)
        return self

    def global_without_plate(self, sub: Plate[T]) -> "Graph[T]":
        """Remove a plate no matter where it is, and all its plates,
        and all its nodes."""
        if sub._graph == self:
            p = sub._parent
            if p is not None:  # This should never happen
                p.without_plate(sub)
        return self

    def with_node(self, node: T) -> "Graph[T]":
        """Add a node to the top level"""
        self._top.with_node(node)
        return self

    def without_node(self, node: T) -> "Graph[T]":
        """Remove a node from the top level"""
        self._top.without_node(node)
        return self

    def global_without_node(self, node: T) -> "Graph[T]":
        """Remove a node no matter where it is"""
        if node in self._nodes:
            self._nodes[node].without_node(node)
        return self

    def with_edge(self, start: T, end: T) -> "Graph[T]":
        if start not in self._nodes:
            self.with_node(start)
        if end not in self._nodes:
            self.with_node(end)
        if start not in self._incoming[end]:
            self._incoming[end].append(start)
        if end not in self._outgoing[start]:
            self._outgoing[start].append(end)
        return self

    def without_edge(self, start: T, end: T) -> "Graph[T]":
        if start in self._nodes and end in self._nodes:
            self._incoming[end].remove(start)
            self._outgoing[start].remove(end)
        return self

    def _is_dag(self, node: T) -> bool:
        if node not in self._nodes:
            return True
        in_flight: List[T] = []
        done: List[T] = []

        def depth_first(current: T) -> bool:
            if current in in_flight:
                return False
            if current in done:
                return True
            in_flight.append(current)
            for child in self._outgoing[current]:
                if not depth_first(child):
                    return False
            in_flight.remove(current)
            done.append(current)
            return True

        return depth_first(node)

    def _dag_hash(self, current: T, map: Dict[T, str]) -> str:
        if current in map:
            return map[current]
        label = self._to_kernel(current)
        children = (self._dag_hash(c, map) for c in self._outgoing[current])
        summary = label + "/".join(sorted(children))

        hash = md5(summary.encode("utf-8")).hexdigest()
        map[current] = hash
        return hash

    def are_dags_isomorphic(self, n1: T, n2: T) -> bool:
        """Determines if two nodes in a graph, which must both be roots of a DAG,
        are isomorphic. Node labels are given by the function, which must return the
        same string for two nodes iff the two nodes are value-equal for the purposes of
        isomorphism detection."""
        map: Dict[T, str] = {}
        assert self._is_dag(n1)
        assert self._is_dag(n2)
        h1 = self._dag_hash(n1, map)
        h2 = self._dag_hash(n2, map)
        return h1 == h2

    def merge_isomorphic(self, n1: T, n2: T) -> bool:
        """Merges two isomorphic nodes.
        Returns true if there was any merge made."""
        # All edges of n2 become edges of n1, and n2 is deleted.
        if n1 not in self._nodes or n2 not in self._nodes:
            return False
        for in_n2 in self._incoming[n2]:
            self.with_edge(in_n2, n1)
        for out_n2 in self._outgoing[n2]:
            self.with_edge(n1, out_n2)
        self.without_node(n2)
        return True

    def merge_isomorphic_many(self, nodes: List[T]) -> bool:
        """Merges a collection of two or more isomorphic nodes into nodes[0]
        Returns true if there was any merge made."""
        result = False
        for i in range(1, len(nodes)):
            result = self.merge_isomorphic(nodes[0], nodes[i]) or result
        return result

    def merge_isomorphic_children(self, node: T) -> bool:
        """Merges all the isomorphic children of a node.
        Returns true if there was any merge made.
        The surviving node is the one with the least name."""
        if node not in self._outgoing:
            return False
        map: Dict[T, str] = {}

        def kernel(n: T) -> str:
            return self._dag_hash(n, map)

        equivalence_classes = partition_by_kernel(self._outgoing[node], kernel)
        result = False
        for eqv in equivalence_classes:
            result = (
                self.merge_isomorphic_many(sorted(eqv, key=self._to_name)) or result
            )
        return result

    def outgoing(self, node: T) -> List[T]:
        if node in self._outgoing:
            return list(self._outgoing[node])
        return []

    def incoming(self, node: T) -> List[T]:
        if node in self._incoming:
            return list(self._incoming[node])
        return []

    def reachable(self, node: T) -> List[T]:
        # Given a node in a graph, return the transitive closure of outgoing
        # nodes, including the original node.
        if node not in self._nodes:
            return []
        in_flight: List[T] = []
        done: List[T] = []

        def depth_first(current: T) -> None:
            if (current not in in_flight) and (current not in done):
                in_flight.append(current)
                for child in self._outgoing[current]:
                    depth_first(child)
                in_flight.remove(current)
                done.append(current)

        depth_first(node)
        return done

    def to_dot(self) -> str:
        """Converts a graph to a program in the DOT language."""
        db: DotBuilder = DotBuilder()

        def add_nodes(sub: Plate[T], name: str) -> None:
            if name != "":
                db.start_subgraph(name, True)
            namer = make_namer(prefix=name + "_")
            for subsub in sub._plates:
                add_nodes(subsub, namer(subsub))
            for n in sub._nodes:
                db.with_node(self._to_name(n), self._to_label(n))
            if name != "":
                db.end_subgraph()

        add_nodes(self._top, "")

        for start, ends in self._outgoing.items():
            for end in ends:
                db.with_edge(self._to_name(start), self._to_name(end))

        return str(db)
