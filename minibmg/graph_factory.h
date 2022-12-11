/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/graph.h"

namespace beanmachine::minibmg {

class Node;

struct NodeId {
  unsigned long value;
  NodeId() : value{0} {}
  explicit NodeId(unsigned long value) : value{value} {}
};
struct ScalarNodeId : public NodeId {
  explicit ScalarNodeId(unsigned long value) : NodeId{value} {}
};
struct ScalarSampleNodeId : public ScalarNodeId {
  explicit ScalarSampleNodeId(unsigned long value) : ScalarNodeId{value} {}
};
struct DistributionNodeId : public NodeId {
  explicit DistributionNodeId(unsigned long value) : NodeId{value} {}
};

} // namespace beanmachine::minibmg

// Make NodeId values usable as a key in a hash table.
template <>
struct std::hash<beanmachine::minibmg::NodeId> {
  std::size_t operator()(const beanmachine::minibmg::NodeId& n) const noexcept {
    return (std::size_t)n.value;
  }
};
template <>
struct std::equal_to<beanmachine::minibmg::NodeId> {
  bool operator()(
      const beanmachine::minibmg::NodeId& lhs,
      const beanmachine::minibmg::NodeId& rhs) const {
    return lhs.value == rhs.value;
  }
};

// Make NodeId values printable using format.
template <>
struct fmt::formatter<beanmachine::minibmg::NodeId>
    : fmt::formatter<std::string> {
  auto format(const beanmachine::minibmg::NodeId& n, format_context& ctx) {
    return formatter<std::string>::format(fmt::format("{}", n.value), ctx);
  }
};

namespace beanmachine::minibmg {

class Graph::Factory {
 public:
  ScalarNodeId constant(double value);
  ScalarNodeId variable(const std::string& name, const int identifier);
  ScalarSampleNodeId sample(
      DistributionNodeId distribution,
      const std::string& rvid = make_fresh_rvid());
  ScalarNodeId add(ScalarNodeId left, ScalarNodeId right);
  ScalarNodeId subtract(ScalarNodeId left, ScalarNodeId right);
  ScalarNodeId negate(ScalarNodeId x);
  ScalarNodeId multiply(ScalarNodeId left, ScalarNodeId right);
  ScalarNodeId divide(ScalarNodeId left, ScalarNodeId right);
  ScalarNodeId pow(ScalarNodeId left, ScalarNodeId right);
  ScalarNodeId exp(ScalarNodeId x);
  ScalarNodeId log(ScalarNodeId x);
  ScalarNodeId atan(ScalarNodeId x);
  ScalarNodeId lgamma(ScalarNodeId x);
  ScalarNodeId polygamma(int n, ScalarNodeId x);
  ScalarNodeId log1p(ScalarNodeId x);
  ScalarNodeId
  if_equal(ScalarNodeId a, ScalarNodeId b, ScalarNodeId c, ScalarNodeId d);
  ScalarNodeId
  if_less(ScalarNodeId a, ScalarNodeId b, ScalarNodeId c, ScalarNodeId d);
  DistributionNodeId normal(ScalarNodeId mean, ScalarNodeId stddev);
  DistributionNodeId half_normal(ScalarNodeId stddev);
  DistributionNodeId beta(ScalarNodeId a, ScalarNodeId b);
  DistributionNodeId bernoulli(ScalarNodeId prob);
  DistributionNodeId exponential(ScalarNodeId rate);

  void observe(ScalarSampleNodeId sample, double value);
  unsigned query(ScalarNodeId value);

  Nodep operator[](const NodeId& node_id) const;
  ScalarNodep operator[](const ScalarNodeId& node_id) const;
  DistributionNodep operator[](const DistributionNodeId& node_id) const;
  ScalarSampleNodep operator[](const ScalarSampleNodeId& node_id) const;

  Graph build();
  ~Factory();

 private:
  Graph::Factory& map = *this; // for convenience
  bool built = false;
  std::unordered_map<NodeId, Nodep> identifer_to_node;
  std::unordered_map<Nodep, NodeId> node_to_identifier;
  std::vector<Nodep> queries;
  std::vector<std::pair<Nodep, double>> observations;
  unsigned long next_identifier = 0;

  ScalarNodeId add_node(ScalarNodep node);
  DistributionNodeId add_node(DistributionNodep node);
  ScalarSampleNodeId add_node(std::shared_ptr<const ScalarSampleNode> node);
  void check_not_built();
};

} // namespace beanmachine::minibmg
