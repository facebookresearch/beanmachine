/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/distribution/distribution.h"

namespace beanmachine {
namespace distribution {

class Product : public Distribution {
 public:
  explicit Product(const std::vector<Node*>& in_nodes);
  Product(
      graph::AtomicType sample_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Product() override {}
  double log_prob(const graph::NodeValue& value) const override;
  void log_prob_iid(const graph::NodeValue& value, Eigen::MatrixXd& log_probs)
      const override;
  // the following is needed to prevent log_prob_iid(value)
  // from being hidden by log_prob_iid(value, log_probs)
  // defined in this class.
  using Distribution::log_prob_iid;
  void gradient_log_prob_value(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;
  void gradient_log_prob_param(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;

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

 private:
  const std::vector<Distribution*> in_distributions;

  void check_required_sample_type_against_sample_type_from_parents(
      graph::AtomicType sample_type);
};

} // namespace distribution
} // namespace beanmachine
