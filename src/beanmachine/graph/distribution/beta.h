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
  void gradient_log_prob_value(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;
  void gradient_log_prob_value(
      const graph::NodeValue& value,
      Eigen::MatrixXd& grad1,
      Eigen::MatrixXd& grad2_diag) const override;
  void gradient_log_prob_param(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;

  void compute_jacobian_hessian(
      const graph::NodeValue& value,
      Eigen::Matrix<double, 1, 2>& jacobian,
      Eigen::Matrix2d& hessian) const;
  void _gradient_log_prob_value(
      const double& val,
      double& grad1,
      double& grad2,
      const double& param_a,
      const double& param_b) const;
};

} // namespace distribution
} // namespace beanmachine
