/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include "beanmachine/graph/util.h"
#include <boost/math/special_functions/polygamma.hpp>
#include <Eigen/Core>
#include <cmath>
#include "beanmachine/graph/graph.h"

namespace beanmachine::util {

// see https://core.ac.uk/download/pdf/41787448.pdf
const double PHI_APPROX_GAMMA = 1.702;

bool approx_zero(double val) {
  return std::abs(val) < graph::PRECISION;
}

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
  if (logprob > 0)
    return true;
  else {
    std::bernoulli_distribution dist(
        std::isnan(logprob) ? 0.0 : std::exp(logprob));
    return dist(gen);
  }
}

bool flip_coin_with_log_prob(std::mt19937& gen, double logprob) {
  return sample_logprob(gen, logprob);
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
  // See "log-sum-exp trick for log-domain calculations" in
  // https://en.wikipedia.org/wiki/LogSumExp
  assert(values.size() != 0);
  double max = *std::max_element(values.begin(), values.end());
  double sum = 0;
  for (auto value : values) {
    sum += std::exp(value - max);
  }
  return std::log(sum) + max;
}

double log_sum_exp(double a, double b) {
  double max_val = a > b ? a : b;
  double sum = std::exp(a - max_val) + std::exp(b - max_val);
  return std::log(sum) + max_val;
}

std::vector<double> probs_given_log_potentials(std::vector<double> log_pot) {
  // p_i = pot_i/Z
  // where Z is the normalization constant sum_i exp(log pot_i).
  // = exp(log(pot_i/Z))
  // = exp(log pot_i - logZ)
  // logZ is log(sum_i exp(log pot_i))
  auto logZ = log_sum_exp(log_pot);
  std::vector<double> probs;
  probs.reserve(log_pot.size());
  for (size_t i = 0; i != log_pot.size(); i++) {
    probs.push_back(std::exp(log_pot[i] - logZ));
  }
  return probs;
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

Eigen::MatrixXd log1pexp(const Eigen::MatrixXd& x) {
  return x.unaryExpr([](double x) { return log1pexp(x); });
}

double log1mexp(double x) {
  assert(x <= 0);
  if (x < -0.693) {
    return std::log1p(-std::exp(x));
  } else {
    return std::log(-std::expm1(x));
  }
}

} // namespace beanmachine::util
