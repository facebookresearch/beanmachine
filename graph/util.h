// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <random>

#include <torch/torch.h>

namespace beanmachine {
namespace util {

bool is_boolean_scalar(const torch::Tensor& value);

bool is_boolean_tensor(const torch::Tensor& value);

bool is_real_scalar(const torch::Tensor& value);

bool is_real_tensor(const torch::Tensor& value);

// sample with probability 1 / (1 + exp(-logodds))
bool sample_logodds(std::mt19937& gen, double logodds);

} // namespace util
} // namespace beanmachine
