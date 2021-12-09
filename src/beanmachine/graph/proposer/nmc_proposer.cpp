/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>
#include <random>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/beta.h"
#include "beanmachine/graph/proposer/delta.h"
#include "beanmachine/graph/proposer/gamma.h"
#include "beanmachine/graph/proposer/mixture.h"
#include "beanmachine/graph/proposer/normal.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/proposer/trunc_cauchy.h"

namespace beanmachine {
namespace proposer {

const double MAIN_PROPOSER_WEIGHT = 1.0;
const double RANDOM_WALK_WEIGHT = 0.01;

std::unique_ptr<Proposer>
nmc_proposer(const graph::NodeValue& value, double grad1, double grad2) {
  bool is_valid_grad = std::isfinite(grad1) && std::isfinite(grad2);
  std::vector<double> weights;
  std::vector<std::unique_ptr<Proposer>> proposers;
  // For boolean variables we will put a point mass on the complementary value
  // and a small mass on the current value. This latter is needed to avoid
  // periodicity.
  if (value.type == graph::AtomicType::BOOLEAN) {
    weights.push_back(0.99);
    proposers.push_back(
        std::make_unique<Delta>(graph::NodeValue(not value._bool)));
    weights.push_back(0.01);
    proposers.push_back(std::make_unique<Delta>(graph::NodeValue(value._bool)));
  }
  // For continuous-valued variables we will mix multiple proposers with various
  // weights with a small probability always given to a random walk proposer.
  if (value.type == graph::AtomicType::PROBABILITY) {
    double x = value._double;
    assert(x > 0 and x < 1);
    // a random walk for a probability is a Beta proposer with strength one
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<Beta>(x, 1 - x));
    // we will approximate a probability variable with a Beta proposer
    // f(x)   = log_prob(x | Beta(a, b))
    // f(x)   = (a-1) log(x) + (b-1) log(1-x) +.. terms in a and b..
    // f'(x)  = (a-1)/x - (b-1)/(1-x)
    // f''(x) = -(a-1)/x^2 - (b-1)/(1-x)^2
    // Solving for a and b.
    // a = 1 - x^2 [-f'(x) + (1-x) f''(x)]
    // b = 1 - (1-x)^2 [f'(x) + x f''(x)]
    double a = 1 - x * x * (-grad1 + (1 - x) * grad2);
    double b = 1 - (1 - x) * (1 - x) * (grad1 + x * grad2);
    if (is_valid_grad and a > 0 and b > 0) {
      weights.push_back(MAIN_PROPOSER_WEIGHT);
      proposers.push_back(std::make_unique<Beta>(a, b));
    }
  } else if (value.type == graph::AtomicType::REAL) {
    double x = value._double;
    // first a random walk
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<Normal>(x, 1.0));
    // we will approximate a real value with a Normal proposer
    // f(x) = log_prob(x | Normal(mu, sigma))
    // f(x) = -log(sigma) -0.5 (x - mu)^2 / sigma^2
    // f'(x) = - (x - mu) / sigma^2
    // f''(x) = - 1 / sigma^2
    // Solving for mu and sigma^2
    // sigma = sqrt(-1 / f''(x) )
    // mu = x - f'(x) / f''(x)
    if (is_valid_grad and grad2 < 0) {
      double sigma = std::sqrt(-1 / grad2);
      double mu = value._double - grad1 / grad2;
      // we will mix multiple proposers with increasing variance and lower
      // probability
      weights.push_back(MAIN_PROPOSER_WEIGHT);
      proposers.push_back(std::make_unique<Normal>(mu, sigma));
      weights.push_back(MAIN_PROPOSER_WEIGHT / 10);
      proposers.push_back(std::make_unique<Normal>(mu, sigma * 10));
    }
  } else if (value.type == graph::AtomicType::POS_REAL) {
    double x = value._double;
    // first a random walk
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<TruncatedCauchy>(value._double, 1.0));
    // we will approximate a positive real value with a truncated cauchy
    // f(x)   = - log(s^2 + (x-m)^2)
    // f'(x)  = - 2(x-m) / (s^2 + (x-m)^2)
    // f''(x) = -2/(s^2 + (x-m)^2) + 4 (x-m)^2 / (s^2 + (x-m)^2)^2
    // Hence: f''(x) / f'(x)^2 =  -0.5 * (s^2 + (x-m)^2)/(x-m)^2 + 1
    //                         =  0.5 * (1 - (s/(x-m))^2)
    // ((x-m)/s)^2 = 1 / (1 - 2 * f''(x) / f'(x)^2)
    // since f'(x) = (-2/s) * ((x-m)/s) / (1 + ((x-m)/s)^2)
    // sgn((x-m)/s) = -sgn(f'(x))
    // s = (-2/f'(x)) * ((x-m)/s) / (1 + ((x-m)/s)^2)
    double scaled_x_sq = 1 / (1 - 2 * grad2 / (grad1 * grad1));
    if (is_valid_grad and scaled_x_sq > 0) {
      double scaled_x = std::sqrt(scaled_x_sq);
      if (grad1 > 0) {
        scaled_x = -scaled_x;
      }
      double scale = (-2 / grad1) * scaled_x / (1 + scaled_x_sq);
      double loc = x - scaled_x * scale;
      // we will mix multiple proposers with increasing variance and lower
      // probability
      weights.push_back(MAIN_PROPOSER_WEIGHT);
      proposers.push_back(std::make_unique<TruncatedCauchy>(loc, scale));
      weights.push_back(MAIN_PROPOSER_WEIGHT / 10.0);
      proposers.push_back(std::make_unique<TruncatedCauchy>(loc, scale * 10));
    }
    // Another random walk is an Exponential distribution centered at the
    // current value
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<Gamma>(1.0, 1.0 / x));
    // we can also approximate a positive value with a Gamma proposer
    // f(x) = log_prob(x | Gamma(alpha, beta))
    // f(x) = alpha*log(beta) + (alpha-1)*log(x) - beta * x - log(G(alpha))
    // f'(x) = (alpha-1)/x - beta
    // f''(x) = -(alpha-1)/x^2
    // Solving for alpha and beta
    // alpha = 1 - x^2 f''(x)
    // beta = - x f''(x) - f'(x)
    double alpha = 1 - x * x * grad2;
    double beta = -x * grad2 - grad1;
    if (is_valid_grad and alpha > 0 and beta > 0) {
      weights.push_back(MAIN_PROPOSER_WEIGHT);
      proposers.push_back(std::make_unique<Gamma>(alpha, beta));
      // another proposer with higher variance
      weights.push_back(MAIN_PROPOSER_WEIGHT / 10);
      proposers.push_back(std::make_unique<Gamma>(alpha / 10, beta / 10));
    }
  }
  return std::make_unique<Mixture>(weights, std::move(proposers));
}

} // namespace proposer
} // namespace beanmachine
