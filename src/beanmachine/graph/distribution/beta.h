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
  graph::AtomicValue sample(std::mt19937& gen) const override;
  void sample(std::mt19937& gen, graph::AtomicValue& sample_value) const override;
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
};

} // namespace distribution
} // namespace beanmachine
