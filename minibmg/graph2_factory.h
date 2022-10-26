/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/graph2.h"

namespace beanmachine::minibmg {

class Node2;

// An opaque identifier for a node.
class Node2Identifier {
 public:
  Node2Identifier() = delete;
  explicit Node2Identifier(unsigned long value);
  unsigned long _value() const;

 private:
  unsigned long value;
};
using Node2Id = std::shared_ptr<Node2Identifier>;

class ScalarNode2Identifier : public Node2Identifier {
 public:
  explicit ScalarNode2Identifier(unsigned long value);
};
using ScalarNode2Id = std::shared_ptr<ScalarNode2Identifier>;

class ScalarSampleNode2Identifier : public ScalarNode2Identifier {
 public:
  explicit ScalarSampleNode2Identifier(unsigned long value);
};
using ScalarSampleNode2Id = std::shared_ptr<ScalarSampleNode2Identifier>;

class DistributionNode2Identifier : public Node2Identifier {
 public:
  explicit DistributionNode2Identifier(unsigned long value);
};
using DistributionNode2Id = std::shared_ptr<DistributionNode2Identifier>;

} // namespace beanmachine::minibmg

// Make Node2Id values usable as a key in a hash table.
template <>
struct std::hash<beanmachine::minibmg::Node2Id> {
  std::size_t operator()(
      const beanmachine::minibmg::Node2Id& n) const noexcept {
    return (std::size_t)n->_value();
  }
};

// Make Node2Id values printable using format.
template <>
struct fmt::formatter<beanmachine::minibmg::Node2Id>
    : fmt::formatter<std::string> {
  auto format(const beanmachine::minibmg::Node2Id& n, format_context& ctx) {
    return formatter<std::string>::format(fmt::format("{}", n->_value()), ctx);
  }
};

namespace beanmachine::minibmg {

class Graph2::Factory {
 public:
  ScalarNode2Id constant(double value);
  ScalarNode2Id variable(const std::string& name, const unsigned identifier);
  ScalarSampleNode2Id sample(
      DistributionNode2Id distribution,
      const std::string& rvid = make_fresh_rvid());
  ScalarNode2Id add(ScalarNode2Id left, ScalarNode2Id right);
  ScalarNode2Id subtract(ScalarNode2Id left, ScalarNode2Id right);
  ScalarNode2Id negate(ScalarNode2Id x);
  ScalarNode2Id multiply(ScalarNode2Id left, ScalarNode2Id right);
  ScalarNode2Id divide(ScalarNode2Id left, ScalarNode2Id right);
  ScalarNode2Id pow(ScalarNode2Id left, ScalarNode2Id right);
  ScalarNode2Id exp(ScalarNode2Id x);
  ScalarNode2Id log(ScalarNode2Id x);
  ScalarNode2Id atan(ScalarNode2Id x);
  ScalarNode2Id lgamma(ScalarNode2Id x);
  ScalarNode2Id polygamma(int n, ScalarNode2Id x);
  ScalarNode2Id
  if_equal(ScalarNode2Id a, ScalarNode2Id b, ScalarNode2Id c, ScalarNode2Id d);
  ScalarNode2Id
  if_less(ScalarNode2Id a, ScalarNode2Id b, ScalarNode2Id c, ScalarNode2Id d);
  DistributionNode2Id normal(ScalarNode2Id mean, ScalarNode2Id stddev);
  DistributionNode2Id half_normal(ScalarNode2Id stddev);
  DistributionNode2Id beta(ScalarNode2Id a, ScalarNode2Id b);
  DistributionNode2Id bernoulli(ScalarNode2Id prob);

  void observe(ScalarNode2Id sample, double value);
  unsigned query(ScalarNode2Id value);

  Node2p operator[](const Node2Id& node_id) const;
  ScalarNode2p operator[](const ScalarNode2Id& node_id) const;
  DistributionNode2p operator[](const DistributionNode2Id& node_id) const;
  ScalarSampleNode2p operator[](const ScalarSampleNode2Id& node_id) const;

  Graph2 build();
  ~Factory();

 private:
  Graph2::Factory& map = *this; // for convenience
  bool built = false;
  std::unordered_map<Node2Id, Node2p> identifer_to_node;
  std::unordered_map<Node2p, Node2Id> node_to_identifier;
  std::vector<Node2p> queries;
  std::list<std::pair<Node2p, double>> observations;
  unsigned long next_identifier = 0;

  ScalarNode2Id add_node(ScalarNode2p node);
  DistributionNode2Id add_node(DistributionNode2p node);
  ScalarSampleNode2Id add_node(std::shared_ptr<ScalarSampleNode2> node);
  void check_not_built();
};

} // namespace beanmachine::minibmg
