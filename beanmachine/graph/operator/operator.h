// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <string>
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

class Operator : public graph::Node {
 public:
  Operator(
      graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Operator() override {}
  bool is_stochastic() const override {
    return op_type == graph::OperatorType::SAMPLE;
  }
  double log_prob() const override;
  void gradient_log_prob(double& grad1, double& grad2) const override;
  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  graph::OperatorType op_type;
};

} // namespace oper
} // namespace beanmachine
