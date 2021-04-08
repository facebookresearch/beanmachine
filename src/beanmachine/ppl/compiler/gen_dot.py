# Copyright (c) Facebook, Inc. and its affiliates.
"""
Visualize the contents of a builder in the DOT graph language.
"""

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problems import fix_problems
from beanmachine.ppl.compiler.graph_labels import get_node_label
from beanmachine.ppl.utils.dotbuilder import DotBuilder


def to_dot(
    bmg: BMGraphBuilder,
    graph_types: bool = False,
    inf_types: bool = False,
    edge_requirements: bool = False,
    after_transform: bool = False,
    label_edges: bool = True,
) -> str:
    """This dumps the entire accumulated graph state, including
    orphans, as a DOT graph description; nodes are enumerated in the order
    they were created."""
    db = DotBuilder()

    if after_transform:
        fix_problems(bmg, bmg._fix_observe_true).raise_errors()
        nodes = bmg._resort_nodes()
    else:
        nodes = bmg.nodes

    max_length = len(str(len(nodes) - 1))

    def to_id(index) -> str:
        return "N" + str(index).zfill(max_length)

    for node, index in nodes.items():
        n = to_id(index)
        node_label = get_node_label(node)
        if graph_types:
            node_label += ":" + node.graph_type.short_name
        if inf_types:
            node_label += ">=" + node.inf_type.short_name
        db.with_node(n, node_label)
        for (i, edge_name, req) in zip(node.inputs, node.edges, node.requirements):
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
