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

bool is_boolean_scalar(const torch::Tensor& value);

bool is_boolean_tensor(const torch::Tensor& value);

bool is_real_scalar(const torch::Tensor& value);

bool is_real_tensor(const torch::Tensor& value);

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

} // namespace util
} // namespace beanmachine
