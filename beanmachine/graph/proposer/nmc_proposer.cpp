// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/proposer/beta.h"
#include "beanmachine/graph/proposer/normal.h"
#include "beanmachine/graph/proposer/mixture.h"

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
  } if (value.type == graph::AtomicType::REAL) {
    // we will approximate a real value with a Normal proposer
    // f(x) = log_prob(x | Normal(mu, sigma))
    // f(x) = -log(sigma) -0.5 (x - mu)^2 / sigma^2
    // f'(x) = - (x - mu) / sigma^2
    // f''(x) = - 1 / sigma^2
    // Solving for mu and sigma^2
    // sigma = sqrt(-1 / f''(x) )
    // mu = x - f'(x) / f''(x)
    // We will use a normal centered at the current value with std 1 if nothing else works
    double mu = value._double;
    double sigma = 1;
    if (grad2 < 0) {
      sigma = std::sqrt(-1 / grad2);
      mu = value._double - grad1 / grad2;
    }
    // we will mix multiple proposers with increasing variance and lower probability
    std::vector<std::unique_ptr<Proposer>> props;
    props.push_back(std::make_unique<Normal>(mu, sigma));
    props.push_back(std::make_unique<Normal>(mu, sigma*10));
    return std::make_unique<Mixture>(std::vector<double>{1, 0.1}, std::move(props));
  }
  // inference shouldn't call this function for other types
  return nullptr;
}

} // namespace proposer
} // namespace beanmachine
