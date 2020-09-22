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
  double log_prob(const graph::AtomicValue& value) const override;
  void gradient_log_prob_value(
      const graph::AtomicValue& value,
      double& grad1,
      double& grad2) const override;
  void gradient_log_prob_value(
      const graph::AtomicValue& value,
      Eigen::MatrixXd& grad1,
      Eigen::MatrixXd& grad2_diag) const override;
  void gradient_log_prob_param(
      const graph::AtomicValue& value,
      double& grad1,
      double& grad2) const override;

  void compute_jacobian_hessian(
      const graph::AtomicValue& value,
      Eigen::Matrix<double, 1, 2>& jacobian,
      Eigen::Matrix2d& hessian) const;
};

} // namespace distribution
} // namespace beanmachine
