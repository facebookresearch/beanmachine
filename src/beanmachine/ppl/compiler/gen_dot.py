# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualize the contents of a builder in the DOT graph language.
"""
from typing import Set

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_requirements import EdgeRequirements
from beanmachine.ppl.compiler.fix_problems import (
    default_skip_optimizations,
    fix_problems,
)
from beanmachine.ppl.compiler.graph_labels import get_edge_labels, get_node_label
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper
from beanmachine.ppl.compiler.sizer import Sizer, size_to_str
from beanmachine.ppl.utils.dotbuilder import DotBuilder


def to_dot(
    bmg: BMGraphBuilder,
    node_types: bool = False,
    node_sizes: bool = False,
    edge_requirements: bool = False,
    after_transform: bool = False,
    label_edges: bool = True,
    skip_optimizations: Set[str] = default_skip_optimizations,
) -> str:
    """This dumps the entire accumulated graph state, including
    orphans, as a DOT graph description; nodes are enumerated in the order
    they were created."""
    lt = LatticeTyper()
    sizer = Sizer()
    reqs = EdgeRequirements(lt)
    db = DotBuilder()

    if after_transform:
        # TODO: It is strange having a visualizer that edits the graph
        # as a side effect, and it is also strange that the only way
        # to visualize the ancestor nodes is to edit the graph.
        #
        # * Remove the after_transform flag; modify the tests so that
        #   tests which currently set after_transform to true instead
        #   call fix_problems first.
        #
        # * Add a whole_graph flag, default to true, which decides
        #   whether to graph the whole thing or not.
        fix_problems(bmg, skip_optimizations).raise_errors()
        node_list = bmg.all_ancestor_nodes()
    else:
        node_list = bmg.all_nodes()

    nodes = {}
    for index, node in enumerate(node_list):
        nodes[node] = index

    max_length = len(str(len(nodes) - 1))

    def to_id(index) -> str:
        return "N" + str(index).zfill(max_length)

    for node, index in nodes.items():
        n = to_id(index)
        node_label = get_node_label(node)
        if node_types:
            node_label += ":" + lt[node].short_name
        if node_sizes:
            node_label += ":" + size_to_str(sizer[node])
        db.with_node(n, node_label)
        for (i, edge_name, req) in zip(
            node.inputs, get_edge_labels(node), reqs.requirements(node)
        ):
            if label_edges:
                edge_label = edge_name
                if edge_requirements:
                    edge_label += ":" + req.short_name
            elif edge_requirements:
                edge_label = req.short_name
            else:
                edge_label = ""

            # Bayesian networks are typically drawn with the arrows
            # in the direction of data flow, not in the direction
            # of dependency.
            start_node = to_id(nodes[i])
            end_node = n
            db.with_edge(start_node, end_node, edge_label)
    return str(db)
