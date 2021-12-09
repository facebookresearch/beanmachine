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

class Tabular : public Distribution {
 public:
  Tabular(
      graph::AtomicType sample_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Tabular() override {}
  bool _bool_sampler(std::mt19937& gen) const override;
  double log_prob(const graph::NodeValue& value) const override;
  void gradient_log_prob_value(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;
  void gradient_log_prob_param(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const override;

 private:
  double get_probability() const;
};

} // namespace distribution
} // namespace beanmachine
