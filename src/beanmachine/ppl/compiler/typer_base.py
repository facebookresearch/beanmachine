# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Once we have accumulated a graph we often need to associate type information
# with some or all of the nodes. This presents a number of practical difficulties:
#
# * The information we wish to associated with a node varies depending on what we
#   are doing with the graph; knowing the BMG type associated with the node might
#   not be relevant for all possible target languages. We therefore wish the type
#   computation logic to admit a variety of different kinds of type information,
#   without excessively duplicating the traversal and update logic described below.
#
# * The type of a given node might depend on the types of all ancestors of that
#   node. However, the set of ancestors of a node might be large, and we might
#   wish to obtain the types of a large number of nodes. We therefore wish to
#   cache as much information as possible to avoid needless recomputation.
#
# * The shortest path of edges between a node and a given ancestor might be very
#   long; longer than the default recursion limit of Python. We therefore require
#   that every algorithm which traverses the graph to compute types be iterative
#   rather than recursive.
#
# * A graph is mutable; we may be using type information to motivate mutations. But
#   since a graph node's type may depend on the type of all its ancestors, when we
#   mutate a node we might have to recompute the types of all its descendents.
#   We wish to do this efficiently, and again, without any recursion.
#
# This abstract base class implements three public operations:
#
# * __getitem__(node)->T  allows you to use the [] accessor to obtain type information
#   about a node. If the type is already cached then it is returned; if not, then the
#   types of all ancestors are computed (if not cached), the node type is computed,
#   cached, and returned.
#
# * __contains__(node)->bool allows you to use the "in" operator to determine if a node's
#   type information has already been cached. (TODO: Is this useful? Maybe remove it.)
#
# * update_type(node)->None informs the type cache that a node has been updated. It
#   recomputes the node's type and efficiently propagates the change to the descendents.
#
# All a typer needs to do is:
#
# * Derive from TyperBase
# * Implement _compute_type_inputs_known to compute the type of a single node.
#
# _compute_type_inputs_known will be called when the type of a node needs to be recomputed.
# It will only be called if the types of all input nodes are known; since typing the inputs
# required typing *their* inputs, we know that all ancestor nodes are typed.

from abc import ABC, abstractmethod
from queue import Queue
from typing import Dict, Generic, TypeVar

import beanmachine.ppl.compiler.bmg_nodes as bn


T = TypeVar("T")


class TyperBase(ABC, Generic[T]):

    _nodes: Dict[bn.BMGNode, T]

    def __init__(self) -> None:
        self._nodes = {}

    def __getitem__(self, node: bn.BMGNode) -> T:
        # If node is already typed, give its type.
        # If not, type it and its inputs if necessary.
        if node not in self._nodes:
            self._update_node_inputs_not_known(node)
            self._propagate_update_to_outputs(node)
        assert node in self._nodes
        return self._nodes[node]

    def __contains__(self, node: bn.BMGNode) -> bool:
        return node in self._nodes

    def _inputs_known(self, node: bn.BMGNode) -> bool:
        return all(i in self._nodes for i in node.inputs)

    def update_type(self, node: bn.BMGNode) -> None:
        # Preconditions:
        #
        # * The node's type might already be known, but might now be wrong.
        # * The types of the inputs of the node might be missing.
        # * All node output types are either not known, because they
        #   are not relevant, or are known but might now be incorrect.
        #
        # Postconditions:
        #
        # * If the node's type was not previously known, it still is not.
        #   Otherwise:
        # * All node input types are known
        # * Node type is correct
        # * Any changes caused by node type being updated have been
        #   propagated to its relevant outputs.
        #

        # If no one previously wanted the type of this node, then there's
        # no need to compute it now, and there are no typed descendants
        # that need updating. Just ignore it, and when someone wants the
        # type, we can compute it then.

        if node not in self._nodes:
            return

        # We have been asked to update the type of a node, presumably
        # because its inputs have been edited. Those inputs might not
        # have been typed in the initial traversal of the graph because
        # they might be new nodes.  Therefore our first task is to
        # determine the types of those inputs and the new type of this
        # node.

        current_type = self._nodes[node]
        self._update_node_inputs_not_known(node)
        new_type = self._nodes[node]
        # We have now computed types for all the previously unknown
        # input types, if there were any, and we have the previous
        # and current type of this node.  If the type of this node
        # changed then the type analysis might be wrong for some
        # of its outputs.  Propagate the change to outputs, and
        # then to their outputs, and so on.
        if current_type != new_type:
            self._propagate_update_to_outputs(node)

    def _propagate_update_to_outputs(self, node: bn.BMGNode) -> None:
        # We've either just typed node for the first time, or its type
        # has just changed. That means that the types of its outputs
        # might have also changed.
        #
        # This propagation should be breadth-first. That is, we should
        # propagate the change to all the outputs, and then all their
        # outputs, and so on.
        #
        # Note that it is possible that a node has an output which is
        # not typed; there might be a branch of the graph which is not
        # an ancestor of any query, observation, sample or factor.
        # We can skip propagating types to such nodes since they are
        # irrelevant for generating the graph.
        #
        # We require that this algorithm, like all algorithms that traverse the
        # graph, be non-recursive.

        work = Queue()
        for o in node.outputs.items:
            if o in self._nodes:
                work.put(o)

        while work.qsize() > 0:
            cur = work.get()
            assert cur in self._nodes
            current_type = self[cur]
            assert self._inputs_known(cur)
            new_type = self._compute_type_inputs_known(cur)
            self._nodes[cur] = new_type
            if current_type == new_type:
                continue
            for o in node.outputs.items:
                if o in self._nodes:
                    work.put(o)

    def _update_node_inputs_not_known(self, node: bn.BMGNode) -> None:
        # Preconditions:
        #
        # * The node is not necessarily already added.
        # * Inputs are not necessarily already added, and similarly with
        #   their inputs and so on.
        #
        # Postconditions:
        #
        # * Transitive closure of untyped inputs is added.
        # * Node is added.
        #
        # We require that this algorithm, like all algorithms that traverse the
        # graph, be non-recursive.

        if node in self._nodes:
            del self._nodes[node]
        work = [node]
        while len(work) > 0:
            cur = work.pop()
            # It is possible that we got the same input in the work
            # stack twice, so this one might already be typed. Just
            # skip it.
            if cur in self._nodes:
                continue
            # We must ensure that inputs are all known. If there are any
            # that are not known, then put the current node back on the
            # work stack and we will come back to it after the inputs are
            # all processed.
            if self._inputs_known(cur):
                self._nodes[cur] = self._compute_type_inputs_known(cur)
            else:
                work.append(cur)
                for i in cur.inputs:
                    if i not in self._nodes:
                        work.append(i)

        assert self._inputs_known(node)
        assert node in self._nodes

    @abstractmethod
    def _compute_type_inputs_known(self, node: bn.BMGNode) -> T:
        pass
