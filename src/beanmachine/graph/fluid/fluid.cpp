/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include "beanmachine/graph/fluid/fluid.h"

namespace beanmachine::graph::fluid {

NodePool remaining_ptrs;

Node* register_ptr(Node* p) {
  remaining_ptrs.insert(p);
  return p;
}

Node* make_node(double d) {
  return register_ptr(new ConstNode(NodeValue(d)));
}

Node* ensure_pos_real(Node* node) {
  Node* result;
  if (node->value.type != AtomicType::POS_REAL) {
    result = register_ptr(new oper::ToPosReal({node}));
  } else {
    result = node;
  }
  return result;
}

Node* ensure_probability(Node* node) {
  Node* result;
  if (node->value.type != AtomicType::PROBABILITY) {
    result = register_ptr(new oper::ToProbability({node}));
  } else {
    result = node;
  }
  return result;
}

Value beta(const Value& a, const Value& b) {
  Node* pos_real_a_node = ensure_pos_real(a.node);
  Node* pos_real_b_node = ensure_pos_real(b.node);
  return register_ptr(new distribution::Beta(
      AtomicType::PROBABILITY, {pos_real_a_node, pos_real_b_node}));
}

Value bernoulli(const Value& p) {
  return register_ptr(new distribution::Bernoulli(
      AtomicType::BOOLEAN, {ensure_probability(p.node)}));
}

Value normal(const Value& mean, const Value& stddev) {
  return register_ptr(new distribution::Normal(
      AtomicType::REAL, {mean.node, ensure_pos_real(stddev.node)}));
}

Value sample(const Value& distribution) {
  return register_ptr(new oper::Sample({distribution.node}));
}

Value operator+(const Value& a, const Value& b) {
  return register_ptr(new oper::Add({a.node, b.node}));
}

Value operator*(const Value& a, const Value& b) {
  return register_ptr(new oper::Multiply({a.node, b.node}));
}

NodeID ensure_node_is_in_graph(Node* node, Graph& graph) {
  for (auto in_node : node->in_nodes) {
    ensure_node_is_in_graph(in_node, graph);
  }
  auto& node_ptrs = graph.node_ptrs();
  auto found = std::find(node_ptrs.begin(), node_ptrs.end(), node);
  if (found == node_ptrs.end()) {
    remaining_ptrs.erase(node);
    return graph.add_node(std::unique_ptr<Node>(node));
  } else {
    return node->index;
  }
}

void observe(const Value& value, const NodeValue& node_value, Graph& graph) {
  auto node_id = ensure_node_is_in_graph(value.node, graph);
  graph.observe(node_id, node_value);
}

void query(const Value& value, Graph& graph) {
  auto node_id = ensure_node_is_in_graph(value.node, graph);
  graph.query(node_id);
}

} // namespace beanmachine::graph::fluid
