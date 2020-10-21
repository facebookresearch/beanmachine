// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <map>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

class Operator : public graph::Node {
 public:
  explicit Operator(graph::OperatorType op_type)
      : graph::Node(graph::NodeType::OPERATOR), op_type(op_type) {}
  ~Operator() override {}
  bool is_stochastic() const override {
    return false;
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
void _gradient_lob_prob(T& first_grad, T& second_grad, Operator const* node) {
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

class OperatorFactory {
 public:
  typedef std::unique_ptr<Operator> (*builder_type)(
      const std::vector<graph::Node*>&);

  OperatorFactory() = delete;
  ~OperatorFactory() {}

  static bool register_op(
      const graph::OperatorType op_type,
      builder_type op_builder);
  static std::unique_ptr<Operator> create_op(
      const graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes);

 private:
  static std::map<int, builder_type>& op_map();
};

} // namespace oper
} // namespace beanmachine
