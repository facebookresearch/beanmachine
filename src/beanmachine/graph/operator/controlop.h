/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace oper {

class IfThenElse : public Operator {
 public:
  explicit IfThenElse(const std::vector<graph::Node*>& in_nodes);
  ~IfThenElse() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<IfThenElse>(in_nodes);
  }

 private:
  static bool is_registered;
};

// This is IfThenElse but the condition is a natural and there are n choices.
class Choice : public Operator {
 public:
  explicit Choice(const std::vector<graph::Node*>& in_nodes);
  ~Choice() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Choice>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
