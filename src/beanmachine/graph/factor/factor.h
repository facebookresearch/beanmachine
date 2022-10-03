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
  explicit Factor(
      graph::FactorType fac_type,
      const std::vector<Node*>& in_nodes)
      : graph::Node(graph::NodeType::FACTOR, in_nodes), fac_type(fac_type) {}
  bool is_stochastic() const override {
    return true;
  }
  void eval(std::mt19937& /* gen */) override {}
  void compute_gradients() override {}
  void backward() override {}

  graph::FactorType fac_type;

  std::unique_ptr<Node> clone() override {
    return new_factor(fac_type, in_nodes);
  }

  virtual std::string to_string() override {
    using namespace std;
    return string(NAMEOF_ENUM(fac_type)) + "(" + in_nodes_string(this) + ")";
  }
};

} // namespace factor
} // namespace beanmachine
