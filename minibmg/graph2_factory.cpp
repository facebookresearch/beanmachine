/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph2_factory.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

Node2Identifier::Node2Identifier(unsigned long value) : value{value} {}

unsigned long Node2Identifier::_value() const {
  return value;
}

ScalarNode2Identifier::ScalarNode2Identifier(unsigned long value)
    : Node2Identifier{value} {}

ScalarSampleNode2Identifier::ScalarSampleNode2Identifier(unsigned long value)
    : ScalarNode2Identifier{value} {}

DistributionNode2Identifier::DistributionNode2Identifier(unsigned long value)
    : Node2Identifier{value} {}

ScalarNode2Id Graph2::Factory::constant(double value) {
  ScalarNode2p result = std::make_shared<ScalarConstantNode2>(value);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::variable(
    const std::string& name,
    const unsigned identifier) {
  ScalarNode2p result = std::make_shared<ScalarVariableNode2>(name, identifier);
  return add_node(result);
}
ScalarSampleNode2Id Graph2::Factory::sample(
    DistributionNode2Id distribution,
    const std::string& rvid) {
  auto result = std::make_shared<ScalarSampleNode2>(map[distribution], rvid);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::add(ScalarNode2Id left, ScalarNode2Id right) {
  ScalarNode2p result = std::make_shared<ScalarAddNode2>(map[left], map[right]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::subtract(
    ScalarNode2Id left,
    ScalarNode2Id right) {
  ScalarNode2p result =
      std::make_shared<ScalarSubtractNode2>(map[left], map[right]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::negate(ScalarNode2Id x) {
  ScalarNode2p result = std::make_shared<ScalarNegateNode2>(map[x]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::multiply(
    ScalarNode2Id left,
    ScalarNode2Id right) {
  ScalarNode2p result =
      std::make_shared<ScalarMultiplyNode2>(map[left], map[right]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::divide(ScalarNode2Id left, ScalarNode2Id right) {
  ScalarNode2p result =
      std::make_shared<ScalarDivideNode2>(map[left], map[right]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::pow(ScalarNode2Id left, ScalarNode2Id right) {
  ScalarNode2p result = std::make_shared<ScalarPowNode2>(map[left], map[right]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::exp(ScalarNode2Id x) {
  ScalarNode2p result = std::make_shared<ScalarExpNode2>(map[x]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::log(ScalarNode2Id x) {
  ScalarNode2p result = std::make_shared<ScalarLogNode2>(map[x]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::atan(ScalarNode2Id x) {
  ScalarNode2p result = std::make_shared<ScalarAtanNode2>(map[x]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::lgamma(ScalarNode2Id x) {
  ScalarNode2p result = std::make_shared<ScalarLgammaNode2>(map[x]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::polygamma(int n, ScalarNode2Id x) {
  ScalarNode2p k = std::make_shared<ScalarConstantNode2>(n);
  ScalarNode2p result = std::make_shared<ScalarPolygammaNode2>(k, map[x]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::if_equal(
    ScalarNode2Id a,
    ScalarNode2Id b,
    ScalarNode2Id c,
    ScalarNode2Id d) {
  ScalarNode2p result =
      std::make_shared<ScalarIfEqualNode2>(map[a], map[b], map[c], map[d]);
  return add_node(result);
}
ScalarNode2Id Graph2::Factory::if_less(
    ScalarNode2Id a,
    ScalarNode2Id b,
    ScalarNode2Id c,
    ScalarNode2Id d) {
  ScalarNode2p result =
      std::make_shared<ScalarIfLessNode2>(map[a], map[b], map[c], map[d]);
  return add_node(result);
}
DistributionNode2Id Graph2::Factory::normal(
    ScalarNode2Id mean,
    ScalarNode2Id stddev) {
  DistributionNode2p result =
      std::make_shared<DistributionNormalNode2>(map[mean], map[stddev]);
  return add_node(result);
}
DistributionNode2Id Graph2::Factory::half_normal(ScalarNode2Id stddev) {
  DistributionNode2p result =
      std::make_shared<DistributionHalfNormalNode2>(map[stddev]);
  return add_node(result);
}
DistributionNode2Id Graph2::Factory::beta(ScalarNode2Id a, ScalarNode2Id b) {
  DistributionNode2p result =
      std::make_shared<DistributionBetaNode2>(map[a], map[b]);
  return add_node(result);
}
DistributionNode2Id Graph2::Factory::bernoulli(ScalarNode2Id prob) {
  DistributionNode2p result =
      std::make_shared<DistributionBernoulliNode2>(map[prob]);
  return add_node(result);
}

void Graph2::Factory::observe(ScalarNode2Id sample, double value) {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  observations.push_back({map[sample], value});
}

ScalarNode2Id Graph2::Factory::add_node(ScalarNode2p node) {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  auto id = std::make_shared<ScalarNode2Identifier>(next_identifier++);
  identifer_to_node[id] = node;
  node_to_identifier[node] = id;
  return id;
}
DistributionNode2Id Graph2::Factory::add_node(DistributionNode2p node) {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  auto id = std::make_shared<DistributionNode2Identifier>(next_identifier++);
  identifer_to_node[id] = node;
  node_to_identifier[node] = id;
  return id;
}
ScalarSampleNode2Id Graph2::Factory::add_node(
    std::shared_ptr<ScalarSampleNode2> node) {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  auto id = std::make_shared<ScalarSampleNode2Identifier>(next_identifier++);
  identifer_to_node[id] = node;
  node_to_identifier[node] = id;
  return id;
}

unsigned Graph2::Factory::query(ScalarNode2Id value) {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  unsigned result = queries.size();
  queries.push_back(map[value]);
  return result;
}

Node2p Graph2::Factory::operator[](const Node2Id& node_id) const {
  return identifer_to_node.at(node_id);
}
ScalarNode2p Graph2::Factory::operator[](const ScalarNode2Id& node_id) const {
  return std::dynamic_pointer_cast<const ScalarNode2>(
      identifer_to_node.at(node_id));
}
DistributionNode2p Graph2::Factory::operator[](
    const DistributionNode2Id& node_id) const {
  return std::dynamic_pointer_cast<const DistributionNode2>(
      identifer_to_node.at(node_id));
}
ScalarSampleNode2p Graph2::Factory::operator[](
    const ScalarSampleNode2Id& node_id) const {
  return std::dynamic_pointer_cast<const ScalarSampleNode2>(
      identifer_to_node.at(node_id));
}

Graph2 Graph2::Factory::build() {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  built = true;
  std::unordered_map<Node2p, Node2p> built_map;
  auto result = Graph2::create(queries, observations, &built_map);

  // Update the node<->identifier maps to reflect the set of dedulplicated
  // nodes in the graph.  This permits the caller to continue using this
  // factory to map node identifiers to nodes in the now deduplicated graph.
  std::unordered_map<Node2Id, Node2p> new_identifer_to_node;
  std::unordered_map<Node2p, Node2Id> new_node_to_identifier;
  for (auto i2n : identifer_to_node) {
    auto found = built_map.find(i2n.second);
    if (found != built_map.end()) {
      new_identifer_to_node[i2n.first] = found->second;
    }
  }
  for (auto n2i : node_to_identifier) {
    auto found = built_map.find(n2i.first);
    if (found != built_map.end()) {
      new_node_to_identifier[found->second] = n2i.second;
    }
  }
  identifer_to_node = new_identifer_to_node;
  node_to_identifier = new_node_to_identifier;

  return result;
}
Graph2::Factory::~Factory() {}

} // namespace beanmachine::minibmg
