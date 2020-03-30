// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/proposer/beta.h"

namespace beanmachine {
namespace proposer {

std::unique_ptr<Proposer> nmc_proposer(const graph::AtomicValue&value, double grad1, double grad2) {
  if (value.type == graph::AtomicType::PROBABILITY) {
    // we will approximate a probability variable with a Beta proposer
    assert(value._double > 0 and value._double < 1);
    // f(x)   = log_prob(x | Beta(a, b))
    // f(x)   = (a-1) log(x) + (b-1) log(1-x) +.. terms in a and b..
    // f'(x)  = (a-1)/x - (b-1)/(1-x)
    // f''(x) = -(a-1)/x^2 - (b-1)/(1-x)^2
    // Solving for a and b.
    // a = 1 - x^2 [-f'(x) + (1-x) f''(x)]
    // b = 1 - (1-x)^2 [f'(x) + x f''(x)]
    double a = 1 - value._double * value._double * (-grad1 + (1-value._double) * grad2);
    double b = 1 - (1-value._double) * (1-value._double) * (grad1 + value._double * grad2);
    // if we can't estimate a or b then we will pick a and b so that the mean is the
    // current value and the strength of the prior is 1.
    if (a <= 0 or b <= 0) {
        a = value._double;
        b = 1 - a;
    }
    return std::make_unique<Beta>(a, b);
  }
  // inference shouldn't call this function for other types
  return nullptr;
}

} // namespace proposer
} // namespace beanmachine
