/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>
#include <algorithm>
#include <random>

namespace beanmachine {
namespace util {

// Check if val is approximately zero
bool approx_zero(double val);

// sample with probability 1 / (1 + exp(-logodds))
bool sample_logodds(std::mt19937& gen, double logodds);

/*
Sample a boolean value given the log of the probability.
:param gen: random number generator
:param logprob: log of probability
:returns: true with probability exp(logprob), false otherwise
*/
bool sample_logprob(std::mt19937& gen, double logprob);

/*
A more intuitive name for :sample_logprob.
:param gen: random number generator
:param logprob: log of probability
:returns: true with probability exp(logprob), false otherwise
*/
bool flip_coin_with_log_prob(std::mt19937& gen, double logprob);

/*
Sample a value from a Beta distribution
:param gen: random number generator
:param a: shape parameter of Beta
:param b: shape parameter of Beta
*/
double sample_beta(std::mt19937& gen, double a, double b);

// compute  1 / (1 + exp(-logodds))
double logistic(double logodds);

/*
Compute the cumulative of the standard Normal upto x.
:param: x
:returns: N(0,1).cdf(x)
*/
double Phi(double x);

/*
Compute the cumulative of the standard Normal upto x.
See https://core.ac.uk/download/pdf/41787448.pdf
:param: x
:returns: N(0,1).cdf(x)  (approximately)
*/
double Phi_approx(double x);

/*
Inverse of the approximate Phi function.
:param z:
:returns x: s.t. Phi_approx(x) == z
*/
double Phi_approx_inv(double z);

/*
Compute the percentiles of a vector of values.
:param values: the vector of values
:param percs: the desired percentiles
:returns: a vector of percentiles
*/
template <typename T>
std::vector<T> percentiles(
    const std::vector<T>& values,
    const std::vector<double>& percs) {
  // copy the values out before sorting them
  std::vector<T> copy(values.begin(), values.end());
  std::sort(copy.begin(), copy.end());
  std::vector<double> result;
  for (auto p : percs) {
    result.push_back(copy[int(copy.size() * p)]);
  }
  return result;
}

/*
Equivalent to log of sum of exponentiations of values,
but more numerically stable.
:param values: vector of log values
:returns: log sum exp of values
*/
double log_sum_exp(const std::vector<double>& values);
double log_sum_exp(double a, double b);

/*
  Given log potentials log pot_i
  where potentials pot_i are an unnormalized probability distribution,
  return the normalized probability distribution p_i.
  p_i = pot_i/Z
  where Z is the normalization constant sum_i exp(log pot_i).
*/
std::vector<double> probs_given_log_potentials(std::vector<double> log_pot);

struct BinaryLogSumExp {
  double operator()(double a, double b) const {
    return log_sum_exp(a, b);
  }
};

/*
Compute the polygamma function. Note n=0 is the digamma function.
:param n:
:returns: polygamma(n, x)
*/
double polygamma(int n, double x);

/*
Compute `log(1 + exp(x))` with numerical stability.
See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
:param x:
:returns: log(1 + exp(x))
*/
double log1pexp(double x);

Eigen::MatrixXd log1pexp(const Eigen::MatrixXd& x);

/*
Compute `log(1 - exp(x))` with numerical stability for negative x values.
See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
:param x:
:returns: log(1 - exp(x))
*/
double log1mexp(double x);

Eigen::MatrixXd log1mexp(const Eigen::MatrixXd& x);

template <typename T>
std::vector<T> make_reserved_vector(size_t n) {
  std::vector<T> result;
  result.reserve(n);
  return result;
}

/*
Computes the log normal density for n idd samples.
Takes the sum of samples as well as the sum of sample squares,
as well as mu (mean) and sigma (standard deviation).
*/
inline double log_normal_density(
    double sum_x,
    double sum_xsq,
    double mu,
    double sigma,
    unsigned n) {
  static const double half_of_log_2_pi = 0.5 * std::log(2 * M_PI);
  return (-std::log(sigma) - half_of_log_2_pi) * n -
      0.5 * (sum_xsq - 2 * mu * sum_x + mu * mu * n) / (sigma * sigma);
}

/*
Computes the log normal density for a sample.
Takes the sampled value
as well as mu (mean) and sigma (standard deviation).
*/
inline double log_normal_density(double x, double mu, double sigma) {
  return log_normal_density(x, x * x, mu, sigma, 1);
}

/*
Computes the log poisson probability for a sample.
Takes the sampled value k
as well as lambda (rate).
*/
inline auto log_poisson_probability(unsigned k, double lambda) {
  return k * std::log(lambda) - lambda - std::lgamma(k + 1);
}

} // namespace util
} // namespace beanmachine
