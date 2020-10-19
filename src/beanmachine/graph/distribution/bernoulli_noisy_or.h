// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/distribution/distribution.h"

namespace beanmachine {
namespace distribution {

/*
BernoulliNoisyOr -- similar to a Bernoulli distribution, but this
is parameterized by a >= 0, s.t. the probability of success = `1 - exp(-a)`.
*/
class BernoulliNoisyOr : public Distribution {
 public:
  BernoulliNoisyOr(
      graph::AtomicType sample_type,
      const std::vector<graph::Node*>& in_nodes);
  ~BernoulliNoisyOr() override {}
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
};

} // namespace distribution
} // namespace beanmachine
