// Copyright (c) Facebook, Inc. and its affiliates.
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

} // namespace oper
} // namespace beanmachine
