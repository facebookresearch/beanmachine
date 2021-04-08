# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this module.
# pyre-ignore-all-errors


from typing import Dict

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.profiler as prof
from beanmachine.graph import Graph
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problems import fix_problems


class GeneratedGraph:
    graph: Graph
    builder: BMGraphBuilder
    node_to_graph_id: Dict[bn.BMGNode, int]
    query_to_query_id: Dict[bn.Query, int]

    def __init__(
        self,
        graph: Graph,
        builder: BMGraphBuilder,
        node_to_graph_id: Dict[bn.BMGNode, int],
        query_to_query_id: Dict[bn.Query, int],
    ) -> None:
        self.graph = graph
        self.builder = builder
        self.node_to_graph_id = node_to_graph_id
        self.query_to_query_id = query_to_query_id


def to_bmg_graph(bmg: BMGraphBuilder) -> GeneratedGraph:
    fix_problems(bmg).raise_errors()
    bmg.pd.begin(prof.build_bmg_graph)
    g = Graph()
    node_to_graph_id: Dict[bn.BMGNode, int] = {}
    query_to_query_id: Dict[bn.Query, int] = {}
    for node in bmg._traverse_from_roots():
        # We add all nodes that are reachable from a query, observation or
        # sample to the BMG graph such that inputs are always added before
        # outputs.
        #
        # TODO: We could consider traversing only nodes reachable from
        # observations or queries.
        #
        # There are four cases to consider:
        #
        # * Observations: there is no associated value returned by the graph
        #   when we add an observation, so there is nothing to track.
        #
        # * Query of a constant: BMG does not support query on a constant.
        #   We skip adding these; when it comes time to fill in the results
        #   dictionary we will just make a vector of the constant value.
        #
        # * Query of an operator: The graph gives us the column index in the
        #   list of samples it returns for this query. We track it in
        #   query_to_query_id.
        #
        # * Any other node: the graph gives us the graph identifier of the new
        #   node. We need to know this for each node that will be used as an input
        #   later, so we track that in node_to_graph_id.

        if isinstance(node, bn.Observation):
            # TODO: Move the add_to_graph logic out of the graph node and into
            # this module. This operation should not be the concern of the
            # node classes.
            node._add_to_graph(g, node_to_graph_id)
        elif isinstance(node, bn.Query):
            if not isinstance(node.operator, bn.ConstantNode):
                query_id = node._add_to_graph(g, node_to_graph_id)
                query_to_query_id[node] = query_id
        else:
            graph_id = node._add_to_graph(g, node_to_graph_id)
            node_to_graph_id[node] = graph_id

    bmg.pd.finish(prof.build_bmg_graph)

    return GeneratedGraph(g, bmg, node_to_graph_id, query_to_query_id)
