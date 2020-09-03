// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <string>
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/distribution/distribution.h"

namespace beanmachine {
namespace oper {

class Operator : public graph::Node {
 public:
  Operator(
      graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Operator() override {}
  bool is_stochastic() const override {
    return op_type == graph::OperatorType::SAMPLE or
        op_type == graph::OperatorType::IID_SAMPLE;
  }
  double log_prob() const override;
  void gradient_log_prob(double& grad1, double& grad2) const override;
  void gradient_log_prob(Eigen::MatrixXd& grad1, Eigen::MatrixXd& grad2_diag)
      const override;
  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  graph::OperatorType op_type;
};

template <class T>
void _gradient_lob_prob(
    T& first_grad,
    T& second_grad,
    Operator const* node) {
  assert(
      node->op_type == graph::OperatorType::SAMPLE or
      node->op_type == graph::OperatorType::IID_SAMPLE);
  const auto dist =
      static_cast<const distribution::Distribution*>(node->in_nodes[0]);
  if (node->grad1 != 0.0) {
    dist->gradient_log_prob_value(node->value, first_grad, second_grad);
  } else {
    dist->gradient_log_prob_param(node->value, first_grad, second_grad);
  }
}

} // namespace oper
} // namespace beanmachine
