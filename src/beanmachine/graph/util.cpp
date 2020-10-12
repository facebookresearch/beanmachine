// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include <boost/math/special_functions/polygamma.hpp>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace util {

// see https://core.ac.uk/download/pdf/41787448.pdf
const double PHI_APPROX_GAMMA = 1.702;

bool sample_logodds(std::mt19937& gen, double logodds) {
  if (logodds < 0) {
    double wt = exp(logodds);
    std::bernoulli_distribution dist(wt / (1 + wt));
    return dist(gen);
  } else {
    double wt = exp(-logodds);
    std::bernoulli_distribution dist(wt / (1 + wt));
    return not dist(gen);
  }
}

bool sample_logprob(std::mt19937& gen, double logprob) {
  std::bernoulli_distribution dist(std::exp(logprob));
  return dist(gen);
}

double sample_beta(std::mt19937& gen, double a, double b) {
  std::gamma_distribution<double> distrib_a(a, 1);
  std::gamma_distribution<double> distrib_b(b, 1);
  double x = distrib_a(gen);
  double y = distrib_b(gen);
  if ((x + y) == 0.0) {
    return graph::PRECISION;
  }
  double p = x / (x + y);
  return p;
}

double logistic(double logodds) {
  return 1.0 / (1.0 + std::exp(-logodds));
}

double Phi(double x) {
  return 0.5 * (1 + std::erf(x / M_SQRT2));
}

double Phi_approx(double x) {
  return 1.0 / (1.0 + std::exp(-PHI_APPROX_GAMMA * x));
}

double Phi_approx_inv(double z) {
  return (std::log(z) - std::log(1 - z)) / PHI_APPROX_GAMMA;
}

double log_sum_exp(const std::vector<double>& values) {
  // find the max and subtract it out
  double max = values[0];
  for (std::vector<double>::size_type idx = 1; idx < values.size(); idx++) {
    if (values[idx] > max) {
      max = values[idx];
    }
  }
  double sum = 0;
  for (auto value : values) {
    sum += std::exp(value - max);
  }
  return std::log(sum) + max;
}

double polygamma(int n, double x) {
  return boost::math::polygamma(n, x);
}

double log1pexp(double x) {
  if (x <= -37) {
    return std::exp(x);
  } else if (x <= 18) {
    return std::log1p(std::exp(x));
  } else if (x <= 33.3) {
    return x + std::exp(-x);
  } else {
    return x;
  }
}

double log1mexp(double x) {
  assert(x <= 0);
  if (x < -0.693) {
    return std::log1p(-std::exp(x));
  } else {
    return std::log(-std::expm1(x));
  }
}

} // namespace util
} // namespace beanmachine
