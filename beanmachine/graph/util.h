// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <random>

#ifndef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif

namespace beanmachine {
namespace util {

// sample with probability 1 / (1 + exp(-logodds))
bool sample_logodds(std::mt19937& gen, double logodds);

/*
Sample a boolean value given the log of the probability.
:param gen: random number generator
:param logprob: log of probability
:returns: true or false
*/
bool sample_logprob(std::mt19937& gen, double logprob);

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
Compute log of the sum of the exponentiation of all the values in the vector
:param values: vector of log values
:returns: log sum exp of values
*/
double log_sum_exp(const std::vector<double>& values);

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

} // namespace util
} // namespace beanmachine
