/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/fluent_factory.h"
#include <memory>

namespace beanmachine::minibmg {

void Graph::FluentFactory::observe(const Traced& sample, double value) {
  if (sample.op() != Operator::SAMPLE) {
    throw std::invalid_argument("can only observe a sample");
  }
  for (auto n : observations) {
    if (n.first == sample.node) {
      throw std::invalid_argument("sample already observed");
    }
  }
  observations.push_back(std::pair(sample.node, value));
}

void Graph::FluentFactory::query(const Traced& value) {
  if (value.node->type != Type::REAL) {
    throw std::invalid_argument("queried node not a value");
  }
  queries.push_back(value.node);
}

Graph Graph::FluentFactory::build() {
  return Graph{queries, observations};
}

Distribution normal(Value mean, Value stddev) {
  return Distribution{std::make_shared<const OperatorNode>(
      std::vector<Nodep>{mean.node, stddev.node},
      Operator::DISTRIBUTION_NORMAL,
      Type::DISTRIBUTION)};
}

Distribution beta(Value a, Value b) {
  return Distribution{std::make_shared<OperatorNode>(
      std::vector<Nodep>{a.node, b.node},
      Operator::DISTRIBUTION_BETA,
      Type::DISTRIBUTION)};
}

Distribution bernoulli(Value p) {
  return Distribution{std::make_shared<OperatorNode>(
      std::vector<Nodep>{p.node},
      Operator::DISTRIBUTION_BERNOULLI,
      Type::DISTRIBUTION)};
}

Value sample(const Distribution& d) {
  return Value{std::make_shared<OperatorNode>(
      std::vector<Nodep>{d.node}, Operator::SAMPLE, Type::REAL)};
}

} // namespace beanmachine::minibmg
