// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/distribution/distribution.h"

namespace beanmachine {
namespace distribution {

class Beta : public Distribution {
 public:
  Beta(
      graph::AtomicType sample_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Beta() override {}
  double _double_sampler(std::mt19937& gen) const override;
  double log_prob(const graph::NodeValue& value) const override;
  void log_prob_iid(const graph::NodeValue& value, Eigen::MatrixXd& log_probs)
      const override;
  void gradient_log_prob_value(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;
  void gradient_log_prob_param(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;
  void compute_jacobian_hessian(
      const graph::NodeValue& value,
      Eigen::Matrix<double, 1, 2>& jacobian,
      Eigen::Matrix2d& hessian) const;

  void backward_value(
      const graph::NodeValue& value,
      graph::DoubleMatrix& back_grad,
      double adjunct = 1.0) const override;
  void backward_value_iid(
      const graph::NodeValue& value,
      graph::DoubleMatrix& back_grad) const override;
  void backward_value_iid(
      const graph::NodeValue& value,
      graph::DoubleMatrix& back_grad,
      Eigen::MatrixXd& adjunct) const override;

  void backward_param(const graph::NodeValue& value, double adjunct = 1.0)
      const override;
  void backward_param_iid(const graph::NodeValue& value) const override;
  void backward_param_iid(
      const graph::NodeValue& value,
      Eigen::MatrixXd& adjunct) const override;

  static void _grad1_log_prob_value(
      double& grad1,
      double x,
      double param_a,
      double param_b);
};

} // namespace distribution
} // namespace beanmachine