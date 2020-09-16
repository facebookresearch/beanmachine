// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace oper {

class StochasticOperator : public Operator {
 public:
  explicit StochasticOperator(graph::OperatorType op_type)
      : Operator(op_type) {}
  ~StochasticOperator() override {}

  void eval(std::mt19937& /* gen */) override {}

  double log_prob() const override {
    return static_cast<const distribution::Distribution*>(in_nodes[0])
        ->log_prob(value);
  }
  void gradient_log_prob(double& first_grad, double& second_grad)
      const override {
    oper::_gradient_lob_prob<double>(first_grad, second_grad, this);
  }
  void gradient_log_prob(
      Eigen::MatrixXd& first_grad,
      Eigen::MatrixXd& second_grad) const override {
    oper::_gradient_lob_prob<Eigen::MatrixXd>(first_grad, second_grad, this);
  }
  bool is_stochastic() const override {
    return true;
  }
  void compute_gradients() override {}
};

class Sample : public oper::StochasticOperator {
 public:
  explicit Sample(const std::vector<graph::Node*>& in_nodes);
  ~Sample() override {}

  void eval(std::mt19937& gen) override;

  static std::unique_ptr<Operator> new_op(const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Sample>(in_nodes);
  }

  static bool is_registered;
};

class IIdSample : public oper::StochasticOperator {
 public:
  explicit IIdSample(const std::vector<graph::Node*>& in_nodes);
  ~IIdSample() override {}

  void eval(std::mt19937& gen) override;

  static std::unique_ptr<Operator> new_op(const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<IIdSample>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
