# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple, Type, Union

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import BMGError, ErrorReport
from beanmachine.ppl.compiler.typer_base import TyperBase


# A "node fixer" is a partial function on nodes; it is similar to a "rule". (See rules.py)
# What distinguishes a node fixer from a rule?
#
# * A node fixer is not an instance of a Rule class; it's just a function.
#
# * A node fixer returns:
#   1. None or Inapplicable if the fixer did not know how to fix the problem
#      TODO: Eliminate use of None as a special return value from a node fixer.
#      Node fixers should return Inapplicable, Fatal, or a node.
#   2. The same node as the input, if the node does not actually need fixing.
#   3. A new node, if the fixer did know how to fix the problem
#   4. Fatal, if the node definitely cannot be fixed, so compilation should cease.
#
#   Note the subtle difference between (1) and (2). Suppose we compose a set of n
#   fixers together, as in the first_match combinator below. If the first fixer
#   returns Inapplicable, then we try the second fixer. If the first fixer returns the
#   input, then that fixer is saying that the node is already correct, and we
#   should not try the second fixer.
#
# * A node fixer mutates an existing graph by adding a new node to it; a Rule just
#   returns a success code containing a new value.
#
# * Rules may be combined together with combinators that apply sub-rules to
#   various branches in a large tree, and the result of such a combination is
#   itself a Rule. Node fixers are combined together to form more complex fixers,
#   but they still just operate on individual nodes. The work of applying node fixers
#   over an entire graph is done by a GraphFixer.


class NodeFixerError:
    pass


Inapplicable = NodeFixerError()
Fatal = NodeFixerError()

NodeFixerResult = Union[bn.BMGNode, None, NodeFixerError]
NodeFixer = Callable[[bn.BMGNode], NodeFixerResult]


def node_fixer_first_match(fixers: List[NodeFixer]) -> NodeFixer:
    def first_match(node: bn.BMGNode) -> NodeFixerResult:
        for fixer in fixers:
            result = fixer(node)
            if result is not None and result is not Inapplicable:
                return result
        return Inapplicable

    return first_match


def type_guard(t: Type, fixer: Callable) -> NodeFixer:
    def guarded(node: bn.BMGNode) -> Optional[bn.BMGNode]:
        return fixer(node) if isinstance(node, t) else None

    return guarded


# A GraphFixer is a function that takes no arguments and returns (1) a bool indicating
# whether the graph fixer made any change or not, and (2) an error report. If the
# error report is non-empty then further processing should stop and the error should
# be reported to the user.

GraphFixerResult = Tuple[bool, ErrorReport]
GraphFixer = Callable[[], GraphFixerResult]

# The identity graph fixer never makes a change or produces an error.
identity_graph_fixer: GraphFixer = lambda: (False, ErrorReport())


def conditional_graph_fixer(condition: bool, fixer: GraphFixer) -> GraphFixer:
    return fixer if condition else identity_graph_fixer


def ancestors_first_graph_fixer(  # noqa
    bmg: BMGraphBuilder,
    typer: TyperBase,
    node_fixer: NodeFixer,
    get_error: Optional[Callable[[bn.BMGNode, int], Optional[BMGError]]] = None,
) -> GraphFixer:
    # Applies the node fixer to each node in the graph builder that is an ancestor,
    # of any sample, query, or observation, starting with ancestors and working
    # towards decendants. Fixes are done one *edge* at a time. That is, when
    # we enumerate a node, we check all its input edges to see if the input node
    # needs to be fixed, and if so, then we update that edge to point from
    # the fixed node to its new output.
    #
    # We enumerate each output node once, but because we then examine each of its
    # input edges, we will possibly encounter the same input node more than once.
    #
    # Rather than rewriting it again, we memoize the result and reuse it.
    # If a fixer indicates a fatally unfixable node then we attempt to report an error
    # describing the problem with the edge. However, we will continue to run fixers
    # on other nodes, hoping that we might report more errors.
    #
    # A typer associates type information with each node in the graph. We have some
    # problems though:
    #
    # * We frequently need to accurately know the type of a node when checking to
    #   see if it needs fixing.
    # * Computing the type of a node requires computing the types of all of its
    #   *ancestor* nodes, which can be quite expensive.
    # * If a mutation changes an input of a node, that node's type might change,
    #   which could then change the types of all of its *descendant* nodes.
    #
    # We solve this performance problem by (1) computing types of nodes on demand
    # and caching the result, (2) being smart about recomputing the type of a node
    # and its descendants when the graph is mutated.  We therefore tell the typer
    # that it needs to re-type a node and its descendants only when a node changes.
    #
    # CONSIDER: Could we use a simpler algorithm here?  That is: for each node,
    # try to fix the node. If successful, remove all the output edges of the old
    # node and add output edges to the new node.  The problem with this approach
    # is that we might end up reporting an error on an edge that is NOT in the
    # subgraph of ancestors of samples, queries and observations, which would be
    # a bad user experience.
    def ancestors_first() -> Tuple[bool, ErrorReport]:
        errors = ErrorReport()
        replacements = {}
        reported = set()
        nodes = bmg.all_ancestor_nodes()
        made_progress = False
        for node in nodes:
            node_was_updated = False
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                # Have we already reported an error on this node? Skip it.
                if c in reported:
                    continue
                # Have we already replaced this input with something?
                # If so, no need to compute the replacement again.
                if c in replacements:
                    if node.inputs[i] is not replacements[c]:
                        node.inputs[i] = replacements[c]
                        node_was_updated = True
                    continue

                replacement = node_fixer(c)

                if isinstance(replacement, bn.BMGNode):
                    replacements[c] = replacement
                    if node.inputs[i] is not replacement:
                        node.inputs[i] = replacement
                        node_was_updated = True
                        made_progress = True
                elif replacement is Fatal:
                    reported.add(c)
                    if get_error is not None:
                        error = get_error(node, i)
                        if error is not None:
                            errors.add_error(error)

            if node_was_updated:
                typer.update_type(node)
        return made_progress, errors

    return ancestors_first


def edge_error_pass(
    bmg: BMGraphBuilder, get_error: Callable[[bn.BMGNode, int], Optional[BMGError]]
) -> GraphFixer:
    """Given a function that takes an edge in the graph and returns an optional error,
    build a pass which checks for errors every edge in the graph that is an ancestor
    of a query, observation, or sample. The edge is given as the descendant node and
    the index of the parent node."""

    def error_pass() -> Tuple[bool, ErrorReport]:
        errors = ErrorReport()
        reported = set()
        nodes = bmg.all_ancestor_nodes()
        for node in nodes:
            for i in range(len(node.inputs)):
                parent = node.inputs[i]
                # We might find errors on many edges, but we only report
                # one error per parent node.
                if parent in reported:
                    continue
                error = get_error(node, i)
                if error is not None:
                    errors.add_error(error)
                    reported.add(parent)
        return False, errors

    return error_pass


def node_error_pass(
    bmg: BMGraphBuilder, get_error: Callable[[bn.BMGNode], Optional[BMGError]]
) -> GraphFixer:
    """Given a function that takes an node in the graph and returns an optional error,
    build a pass which checks for errors every node in the graph that is an ancestor
    of a query, observation, or sample."""

    def error_pass() -> Tuple[bool, ErrorReport]:
        errors = ErrorReport()
        nodes = bmg.all_ancestor_nodes()
        for node in nodes:
            error = get_error(node)
            if error is not None:
                errors.add_error(error)
        return False, errors

    return error_pass


def sequential_graph_fixer(fixers: List[GraphFixer]) -> GraphFixer:
    """Takes a list of graph fixers and applies each in turn once unless one fails."""

    def sequential() -> GraphFixerResult:
        made_progress = False
        errors = ErrorReport()
        for fixer in fixers:
            fixer_made_progress, errors = fixer()
            made_progress |= fixer_made_progress
            if errors.any():
                break
        return made_progress, errors

    return sequential


def fixpoint_graph_fixer(fixer: GraphFixer) -> GraphFixer:
    """Executes a graph fixer repeatedly until it stops making progress
    or produces an error."""

    def fixpoint() -> GraphFixerResult:
        while True:
            made_progress, errors = fixer()
            if not made_progress or errors.any():
                return made_progress, errors

    return fixpoint


# TODO: Create a fixpoint combinator on GraphFixers.
