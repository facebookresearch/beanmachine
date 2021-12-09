/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/factor/factor.h"

namespace beanmachine {
namespace factor {

// A factor that exponentiates a product of potentials or probabilities
class ExpProduct : public Factor {
 public:
  explicit ExpProduct(const std::vector<graph::Node*>& in_nodes);
  ~ExpProduct() override {}
  double log_prob() const override;
  void gradient_log_prob(
      const graph::Node* target_node,
      double& grad1,
      double& grad2) const override;
  void backward() override;
};

} // namespace factor
} // namespace beanmachine
