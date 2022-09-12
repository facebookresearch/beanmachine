/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <boost/math/special_functions/polygamma.hpp>
#include <stdexcept>

#include "beanmachine/graph/global/nuts.h"

#include <Eigen/Core>
#include <iostream>
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace util {

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

Eigen::MatrixXd log1mexp(const Eigen::MatrixXd& x) {
  return x.unaryExpr([](double x) { return log1mexp(x); });
}

double compute_mean_at_index(
    std::vector<std::vector<graph::NodeValue>> samples,
    std::size_t index) {
  double mean = 0;
  for (size_t i = 0; i < samples.size(); i++) {
    assert(samples[i].size() > index);
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    mean += samples[i][index]._double;
  }
  mean /= samples.size();
  return mean;
}

std::vector<double> compute_means(
    std::vector<std::vector<graph::NodeValue>> samples) {
  if (samples.empty()) {
    return std::vector<double>();
  }
  auto num_dims = samples[0].size();
  auto means = std::vector<double>(num_dims);
  for (size_t i = 0; i != num_dims; i++) {
    means[i] = compute_mean_at_index(samples, i);
  }
  return means;
}

void test_nmc_against_nuts(
    graph::Graph& graph,
    int num_rounds,
    int num_samples,
    int warmup_samples,
    std::function<unsigned()> seed_getter,
    std::function<void(double, double)> tester) {
  using namespace std;
  if (graph.queries.empty()) {
    throw invalid_argument(
        "test_nmc_against_nuts requires at least one query in graph.");
  }
  auto measured_max_abs_mean_diff = 0.0;
  for (int i = 0; i != num_rounds; i++) {
    auto seed = seed_getter();

    auto means_nmc =
        graph.infer_mean(num_samples, graph::InferenceType::NMC, seed);

    graph::NUTS nuts = graph::NUTS(graph);
    auto samples = nuts.infer(num_samples, seed, warmup_samples);
    auto means_nuts = compute_means(samples);

    assert(!means_nmc.empty());
    assert(!means_nuts.empty());

    tester(means_nmc[0], means_nuts[0]);

    auto abs_diff = std::abs(means_nmc[0] - means_nuts[0]);
    if (abs_diff > measured_max_abs_mean_diff) {
      measured_max_abs_mean_diff = abs_diff;
    }

    cout << "NMC  result: " << means_nmc[0] << endl;
    cout << "NUTS result: " << means_nuts[0] << endl;
  }
  cout << "Measured max absolute difference: " << measured_max_abs_mean_diff
       << endl;
}

} // namespace util
} // namespace beanmachine
