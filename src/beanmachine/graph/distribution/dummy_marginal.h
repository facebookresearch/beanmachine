/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/marginalization/subgraph.h"

namespace beanmachine {
namespace distribution {

class DummyMarginal : public Distribution {
 public:
  explicit DummyMarginal(std::unique_ptr<graph::SubGraph> subgraph_ptr);
  ~DummyMarginal() override {}
  double log_prob(const graph::NodeValue& /*value*/) const override {
    return 0.0;
  }
  void log_prob_iid(
      const graph::NodeValue& /* value */,
      Eigen::MatrixXd& /* log_probs */) const override {}
  void gradient_log_prob_value(
      const graph::NodeValue& /*value*/,
      double& /*grad1*/,
      double& /*grad2*/) const override {}
  void gradient_log_prob_param(
      const graph::NodeValue& /*value*/,
      double& /*grad1*/,
      double& /*grad2*/) const override {}
  void backward_value(
      const graph::NodeValue& /*value*/,
      graph::DoubleMatrix& /*back_grad*/,
      double /*adjunct*/ = 1.0) const override {}
  void backward_param(
      const graph::NodeValue& /*value*/,
      double /*adjunct*/ = 1.0) const override {}
  std::unique_ptr<graph::SubGraph> subgraph_ptr;
};

} // namespace distribution
} // namespace beanmachine
