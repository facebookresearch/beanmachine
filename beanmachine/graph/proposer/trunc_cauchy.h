// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

class TruncatedCauchy : public Proposer {
 public:
  /*
  Constructor for TruncatedCauchy class (truncated to be positive).
  pdf(x) proportional to  (1/scale) * (1/(1 + ((x-loc)/scale)^2))
  :param alpha: loc
  :param beta: scale
  */
  TruncatedCauchy(double loc, double scale);
  /*
  Sample a value from the proposer.
  :param gen: Random number generator.
  :returns: A value.
  */
  graph::AtomicValue sample(std::mt19937& gen) const override;
  /*
  Compute the log_prob of a value.
  :param value: The value to evaluate the distribution.
  :returns: log probability of value.
  */
  double log_prob(graph::AtomicValue& value) const override;
 private:
  double loc;
  double scale;
  double atan_0; // arctan ( (0-loc) / scale)
  // log_prob = log_pdf_constant + log(1/scale) + log(1/(1+((x-loc)/scale)^2))
  double log_pdf_constant;
};

} // namespace distribution
} // namespace beanmachine
