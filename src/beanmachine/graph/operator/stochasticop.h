/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/transform/transform.h"

namespace beanmachine {
namespace oper {

class StochasticOperator : public Operator {
 public:
  explicit StochasticOperator(graph::OperatorType op_type)
      : Operator(op_type), transform_type(graph::TransformType::NONE) {}
  ~StochasticOperator() override {}

  void eval(std::mt19937& gen) override {
    const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);
    dist->sample(gen, value);
  }
  double log_prob() const override {
    return static_cast<const distribution::Distribution*>(in_nodes[0])
        ->log_prob(value);
  }
  void gradient_log_prob(
      const graph::Node* target_node,
      double& first_grad,
      double& second_grad) const override;
  bool is_stochastic() const override {
    return true;
  }

  void compute_gradients() override {}
  // TODO: compute_gradients is not doing anything currently,
  // but this is not the correct thing to do in all contexts.
  // For some situations we do need the gradients of
  // stochastic functions.
  // See for example https://pytorch.org/docs/stable/distributions.html
  // and https://arxiv.org/abs/1506.05254.
  // Because the Mixture distribution is one such case,
  // I (Rodrigo) plan (as of May 2022) to introduce some such gradients
  // using reparametrization (as described in these sources)
  // while throwing errors in non-supported cases.

  void backward() override {
    _backward(true);
  }
  virtual void _backward(bool /* skip_observed */) {}

  graph::NodeValue* get_original_value(bool sync_from_unconstrained);
  graph::NodeValue* get_unconstrained_value(bool sync_from_constrained);
  double log_abs_jacobian_determinant();
  graph::DoubleMatrix* get_unconstrained_gradient();

  graph::NodeValue unconstrained_value;
  graph::TransformType transform_type;
  graph::Transformation* transform;
};

class Sample : public oper::StochasticOperator {
 public:
  explicit Sample(const std::vector<graph::Node*>& in_nodes);
  ~Sample() override {}
  void _backward(bool skip_observed) override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Sample>(in_nodes);
  }

  static bool is_registered;
};

class IIdSample : public oper::StochasticOperator {
 public:
  explicit IIdSample(const std::vector<graph::Node*>& in_nodes);
  ~IIdSample() override {}
  void _backward(bool skip_observed) override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<IIdSample>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
