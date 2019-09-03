// Copyright (c) Facebook, Inc. and its affiliates.
#include "util.h"

namespace beanmachine {
namespace util {

bool is_boolean_scalar(const torch::Tensor& value) {
  if (value.dim() != 0 or
      value.type().scalarType() != torch::ScalarType::Byte or
      value.item<uint8_t>() > 1) {
    return false;
  } else {
    return true;
  }
}

bool is_boolean_tensor(const torch::Tensor& value) {
  return (
      value.type().scalarType() == torch::ScalarType::Byte and
      value.le(1).all().item<uint8_t>());
}

bool is_real_tensor(const torch::Tensor& value) {
  torch::ScalarType sctype = value.type().scalarType();
  return (
      sctype == torch::ScalarType::Half or sctype == torch::ScalarType::Float or
      sctype == torch::ScalarType::Double);
}

bool is_real_scalar(const torch::Tensor& value) {
  torch::ScalarType sctype = value.type().scalarType();
  return (
      value.dim() == 0 and
      (sctype == torch::ScalarType::Half or
       sctype == torch::ScalarType::Float or
       sctype == torch::ScalarType::Double));
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

} // namespace util
} // namespace beanmachine
