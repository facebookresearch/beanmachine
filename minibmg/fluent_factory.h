/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// For values
using Value = Traced;

// For distributions
class FluentDistribution {
 public:
  Nodep node;
  /* implicit */ FluentDistribution(Nodep node) : node{node} {
    if (node->type != Type::DISTRIBUTION) {
      throw std::invalid_argument("node is not a value");
    }
  }
  inline Operator op() const {
    return node->op;
  }
};

FluentDistribution half_normal(Value stddev);

FluentDistribution normal(Value mean, Value stddev);

FluentDistribution beta(Value a, Value b);

FluentDistribution bernoulli(Value p);

Value sample(const FluentDistribution& d, std::string rvid = make_fresh_rvid());

class Graph::FluentFactory {
 public:
  void observe(const Traced& sample, double value);
  void query(const Traced& value);
  Graph build();

 private:
  std::vector<Nodep> queries;
  std::list<std::pair<Nodep, double>> observations;
};

} // namespace beanmachine::minibmg
