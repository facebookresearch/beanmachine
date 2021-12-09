/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace transform {

class Log : public graph::Transformation {
 public:
  Log() : Transformation(graph::TransformType::LOG) {}
  ~Log() override {}

  void operator()(
      const graph::NodeValue& constrained,
      graph::NodeValue& unconstrained) override;
  void inverse(
      graph::NodeValue& constrained,
      const graph::NodeValue& unconstrained) override;
  double log_abs_jacobian_determinant(
      const graph::NodeValue& constrained,
      const graph::NodeValue& unconstrained) override;
  void unconstrained_gradient(
      graph::DoubleMatrix& back_grad,
      const graph::NodeValue& constrained,
      const graph::NodeValue& unconstrained) override;
};

} // namespace transform
} // namespace beanmachine
