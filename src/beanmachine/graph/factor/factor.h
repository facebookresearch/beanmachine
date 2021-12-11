/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <string>
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace factor {

class Factor : public graph::Node {
 public:
  static std::unique_ptr<Factor> new_factor(
      graph::FactorType fac_type,
      const std::vector<graph::Node*>& in_nodes);
  explicit Factor(graph::FactorType fac_type)
      : graph::Node(graph::NodeType::FACTOR), fac_type(fac_type) {}
  bool is_stochastic() const override {
    return true;
  }
  void eval(std::mt19937& /* gen */) override {}
  void compute_gradients() override {}
  void backward() override {}

  graph::FactorType fac_type;
};

} // namespace factor
} // namespace beanmachine
