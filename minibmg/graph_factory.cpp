/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_factory.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

ScalarNodeId Graph::Factory::constant(double value) {
  ScalarNodep result = std::make_shared<const ScalarConstantNode>(value);
  return add_node(result);
}
ScalarNodeId Graph::Factory::variable(
    const std::string& name,
    const int identifier) {
  ScalarNodep result =
      std::make_shared<const ScalarVariableNode>(name, identifier);
  return add_node(result);
}
ScalarSampleNodeId Graph::Factory::sample(
    DistributionNodeId distribution,
    const std::string& rvid) {
  auto result =
      std::make_shared<const ScalarSampleNode>(map[distribution], rvid);
  return add_node(result);
}
ScalarNodeId Graph::Factory::add(ScalarNodeId left, ScalarNodeId right) {
  ScalarNodep result =
      std::make_shared<const ScalarAddNode>(map[left], map[right]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::subtract(ScalarNodeId left, ScalarNodeId right) {
  ScalarNodep result =
      std::make_shared<const ScalarSubtractNode>(map[left], map[right]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::negate(ScalarNodeId x) {
  ScalarNodep result = std::make_shared<const ScalarNegateNode>(map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::multiply(ScalarNodeId left, ScalarNodeId right) {
  ScalarNodep result =
      std::make_shared<const ScalarMultiplyNode>(map[left], map[right]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::divide(ScalarNodeId left, ScalarNodeId right) {
  ScalarNodep result =
      std::make_shared<const ScalarDivideNode>(map[left], map[right]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::pow(ScalarNodeId left, ScalarNodeId right) {
  ScalarNodep result =
      std::make_shared<const ScalarPowNode>(map[left], map[right]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::exp(ScalarNodeId x) {
  ScalarNodep result = std::make_shared<const ScalarExpNode>(map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::log(ScalarNodeId x) {
  ScalarNodep result = std::make_shared<const ScalarLogNode>(map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::atan(ScalarNodeId x) {
  ScalarNodep result = std::make_shared<const ScalarAtanNode>(map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::lgamma(ScalarNodeId x) {
  ScalarNodep result = std::make_shared<const ScalarLgammaNode>(map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::polygamma(int n, ScalarNodeId x) {
  ScalarNodep k = std::make_shared<const ScalarConstantNode>(n);
  ScalarNodep result = std::make_shared<const ScalarPolygammaNode>(k, map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::log1p(ScalarNodeId x) {
  ScalarNodep result = std::make_shared<const ScalarLog1pNode>(map[x]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::if_equal(
    ScalarNodeId a,
    ScalarNodeId b,
    ScalarNodeId c,
    ScalarNodeId d) {
  ScalarNodep result =
      std::make_shared<const ScalarIfEqualNode>(map[a], map[b], map[c], map[d]);
  return add_node(result);
}
ScalarNodeId Graph::Factory::if_less(
    ScalarNodeId a,
    ScalarNodeId b,
    ScalarNodeId c,
    ScalarNodeId d) {
  ScalarNodep result =
      std::make_shared<const ScalarIfLessNode>(map[a], map[b], map[c], map[d]);
  return add_node(result);
}
DistributionNodeId Graph::Factory::normal(
    ScalarNodeId mean,
    ScalarNodeId stddev) {
  DistributionNodep result =
      std::make_shared<DistributionNormalNode>(map[mean], map[stddev]);
  return add_node(result);
}
DistributionNodeId Graph::Factory::half_normal(ScalarNodeId stddev) {
  DistributionNodep result =
      std::make_shared<DistributionHalfNormalNode>(map[stddev]);
  return add_node(result);
}
DistributionNodeId Graph::Factory::beta(ScalarNodeId a, ScalarNodeId b) {
  DistributionNodep result =
      std::make_shared<DistributionBetaNode>(map[a], map[b]);
  return add_node(result);
}
DistributionNodeId Graph::Factory::bernoulli(ScalarNodeId prob) {
  DistributionNodep result =
      std::make_shared<DistributionBernoulliNode>(map[prob]);
  return add_node(result);
}

DistributionNodeId Graph::Factory::exponential(ScalarNodeId rate) {
  DistributionNodep result =
      std::make_shared<DistributionExponentialNode>(map[rate]);
  return add_node(result);
}

void Graph::Factory::check_not_built() {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
}

void Graph::Factory::observe(ScalarSampleNodeId sample, double value) {
  check_not_built();
  observations.push_back({map[sample], value});
}

ScalarNodeId Graph::Factory::add_node(ScalarNodep node) {
  check_not_built();
  auto id = ScalarNodeId{next_identifier++};
  identifer_to_node[id] = node;
  node_to_identifier[node] = id;
  return id;
}
DistributionNodeId Graph::Factory::add_node(DistributionNodep node) {
  check_not_built();
  auto id = DistributionNodeId{next_identifier++};
  identifer_to_node[id] = node;
  node_to_identifier[node] = id;
  return id;
}
ScalarSampleNodeId Graph::Factory::add_node(
    std::shared_ptr<const ScalarSampleNode> node) {
  check_not_built();
  auto id = ScalarSampleNodeId{next_identifier++};
  identifer_to_node[id] = node;
  node_to_identifier[node] = id;
  return id;
}

unsigned Graph::Factory::query(ScalarNodeId value) {
  check_not_built();
  unsigned result = queries.size();
  queries.push_back(map[value]);
  return result;
}

Nodep Graph::Factory::operator[](const NodeId& node_id) const {
  return identifer_to_node.at(node_id);
}
ScalarNodep Graph::Factory::operator[](const ScalarNodeId& node_id) const {
  return downcast<ScalarNode>(identifer_to_node.at(node_id));
}
DistributionNodep Graph::Factory::operator[](
    const DistributionNodeId& node_id) const {
  return downcast<DistributionNode>(identifer_to_node.at(node_id));
}
ScalarSampleNodep Graph::Factory::operator[](
    const ScalarSampleNodeId& node_id) const {
  return downcast<ScalarSampleNode>(identifer_to_node.at(node_id));
}

Graph Graph::Factory::build() {
  check_not_built();
  built = true;
  std::unordered_map<Nodep, Nodep> built_map;
  auto result = Graph::create(queries, observations, &built_map);

  // Update the node<->identifier maps to reflect the set of deduplicated
  // nodes in the graph.  This permits the caller to continue using this
  // factory to map node identifiers to nodes in the now deduplicated graph.
  std::unordered_map<NodeId, Nodep> new_identifer_to_node;
  std::unordered_map<Nodep, NodeId> new_node_to_identifier;
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
Graph::Factory::~Factory() {}

} // namespace beanmachine::minibmg
