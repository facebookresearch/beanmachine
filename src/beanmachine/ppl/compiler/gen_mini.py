# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder

_node_type_to_distribution = {
    bn.BernoulliNode: "DISTRIBUTION_BERNOULLI",
    bn.BetaNode: "DISTRIBUTION_BETA",
    bn.NormalNode: "DISTRIBUTION_NORMAL",
}

_node_type_to_operator = {
    bn.MultiplicationNode: "MULTIPLY",
    bn.SampleNode: "SAMPLE",
}

MiniNode = Dict[str, Any]


class ToMini:
    _node_to_mini_node: Dict[bn.BMGNode, MiniNode]
    _observed_constants: int
    _queries: int
    _mini_nodes: List[MiniNode]

    def __init__(self) -> None:
        self._node_to_mini_node = {}
        self._observed_constants = 0
        self._queries = 0
        self._mini_nodes = []

    def to_json(self, bmg: BMGraphBuilder, indent=None) -> str:
        self._observed_constants = 0
        self._queries = 0
        self._mini_nodes = []
        self._node_to_mini_node = {}
        # all_ancestor_nodes enumerates all samples, queries,
        # observations, and all of their ancestors, in topo-sorted
        # order, parents before children.
        for node in bmg.all_ancestor_nodes():
            self._add_node_to_mini_nodes(node)
        mini = {
            "comment": "Mini BMG",
            "nodes": self._mini_nodes,
        }
        return json.dumps(mini, indent=indent)

    def _add_mini_node(self, mini: MiniNode) -> None:
        mini["sequence"] = len(self._mini_nodes)
        self._mini_nodes.append(mini)

    def _node_to_mini_seq(self, node: bn.BMGNode) -> int:
        return self._node_to_mini_node[node]["sequence"]

    def _add_inputs(self, mini: MiniNode, node: bn.BMGNode) -> None:
        in_nodes = [self._node_to_mini_seq(i) for i in node.inputs]
        if len(in_nodes) > 0:
            mini["in_nodes"] = in_nodes

    def _make_query(self, node: bn.Query) -> MiniNode:
        mini: MiniNode = {
            "operator": "QUERY",
            "type": "NONE",
            "query_index": self._queries,
        }
        self._queries += 1
        self._add_inputs(mini, node)
        return mini

    def _make_constant(self, value: Any) -> MiniNode:
        return {
            "operator": "CONSTANT",
            "type": "REAL",
            "value": float(value),  # TODO: Deal with tensors
        }

    def _make_distribution(self, node: bn.DistributionNode) -> MiniNode:
        op = _node_type_to_distribution[type(node)]  # pyre-ignore
        mini: MiniNode = {
            "operator": op,
            "type": "DISTRIBUTION",
        }
        self._add_inputs(mini, node)
        return mini

    def _make_operator(self, node: bn.OperatorNode) -> MiniNode:
        op = _node_type_to_operator[type(node)]  # pyre-ignore
        mini: Dict[str, Any] = {
            "operator": op,
            "type": "REAL",
        }
        self._add_inputs(mini, node)
        return mini

    def _is_observed_sample(self, node: bn.BMGNode) -> bool:
        return isinstance(node, bn.SampleNode) and any(
            isinstance(o, bn.Observation) for o in node.outputs.items
        )

    def _get_sample_observation(self, node: bn.SampleNode) -> Any:
        for o in node.outputs.items:
            if isinstance(o, bn.Observation):
                return o.value
        return None

    def _make_observed_sample(self, node: bn.SampleNode) -> None:
        # Various parts of our system handle observations slightly
        # differently, which can be somewhat confusing. Here is how
        # it works:
        #
        # * In the graph accumulator, an observation is a node whose
        #   parent is a sample node, and which contains a constant.
        #
        # * In BMG, an observation is not a node. Rather, it is an
        #   annotation associating a value with a sample node.
        #
        # * In MiniBMG, both the observation and the observed value
        #   are nodes, and observations are parented by a distribution,
        #   not by a sample. In this system, observations are essentially
        #   a special kind of sample that has two parents: a distribution
        #   and a value.
        #
        # How then will we transform the accumulated graph into MiniBMG?
        #
        # Suppose we have accumulated this graph, for arbitrary subgraphs
        # X and Y:
        #
        #             2
        #            / \
        #            BETA
        #              |   \
        #          SAMPLE   QUERY
        #              |
        #         BERNOULLI
        #         /        \
        #       SAMPLE    SAMPLE
        #      /       \      |
        #   OBSERVE(0)  X     Y
        #
        # then we will emit this MiniBMG:
        #
        #             2
        #            / \
        #            BETA
        #              |   \
        #          SAMPLE   QUERY
        #              |
        #    0    BERNOULLI
        #   / \   /      \
        #  X   OBSERVE   SAMPLE
        #                 |
        #                 Y
        #
        # Note that subgraph X has been re-parented to the constant,
        # and the OBSERVE is also a child of the constant.
        #
        # We return None here because this code already takes care
        # of ensuring that the new nodes are added to the list,
        # that a sequence id is generated, and that the observed
        # sample is mapped to the mini const.

        ob = self._get_sample_observation(node)
        mini_const = self._make_constant(ob)
        self._add_mini_node(mini_const)
        const_seq = mini_const["sequence"]
        dist_seq = self._node_to_mini_seq(node.operand)
        in_nodes = [dist_seq, const_seq]
        mini_obs = {
            "operator": "OBSERVE",
            "type": "NONE",
            "in_nodes": in_nodes,
        }
        self._add_mini_node(mini_obs)
        self._node_to_mini_node[node] = mini_const

    def _add_node_to_mini_nodes(self, node: bn.BMGNode) -> None:
        mini: Optional[MiniNode] = None
        if self._is_observed_sample(node):
            # We have special handling for observed queries.
            # mini stays None because the special handler ensures
            # that the maps are set up correctly.
            assert isinstance(node, bn.SampleNode)
            self._make_observed_sample(node)
        elif isinstance(node, bn.Observation):
            # We do nothing when we encounter an observation node.
            # Rather, the observation is added to MiniBMG state
            # when the observed sample is handled.
            pass
        elif isinstance(node, bn.Query):
            mini = self._make_query(node)
        elif isinstance(node, bn.ConstantNode):
            mini = self._make_constant(node.value)
        elif isinstance(node, bn.DistributionNode):
            mini = self._make_distribution(node)
        elif isinstance(node, bn.OperatorNode):
            mini = self._make_operator(node)
        else:
            raise ValueError(f"{type(node)} is not supported by miniBMG")

        if mini is not None:
            self._add_mini_node(mini)
            self._node_to_mini_node[node] = mini


def to_mini(bmg: BMGraphBuilder, indent=None) -> str:
    # TODO: Run an error checking pass that rejects nodes
    # we cannot map to Mini BMG
    return ToMini().to_json(bmg, indent=indent)
