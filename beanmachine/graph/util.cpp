// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/util.h"

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

bool sample_logprob(std::mt19937& gen, double logprob) {
  std::bernoulli_distribution dist(std::exp(logprob));
  return dist(gen);
}

double sample_beta(std::mt19937& gen, double a, double b) {
  std::gamma_distribution<double> distrib_a(a, 1);
  std::gamma_distribution<double> distrib_b(b, 1);
  double x = distrib_a(gen);
  double y = distrib_b(gen);
  double p =  x / (x + y);
  return p;
}

double logistic(double logodds) {
  return 1.0 / (1.0 + exp(-logodds));
}

} // namespace util
} // namespace beanmachine
