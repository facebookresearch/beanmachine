// Copyright (c) Facebook, Inc. and its affiliates.
#include <string>
#include <random>
#include <cmath>

#include "beanmachine/graph/distribution/half_cauchy.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

HalfCauchy::HalfCauchy(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::HALF_CAUCHY, sample_type) {
  // a HalfCauchy distribution has one parent a scale which is positive real
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "HalfCauchy distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "HalfCauchy parent must be positive real number");
  }
  // only positive real-valued samples are possible
  if (sample_type != AtomicType::POS_REAL) {
    throw std::invalid_argument("HalfCauchy distribution produces positive real number samples");
  }
}

AtomicValue HalfCauchy::sample(std::mt19937& gen) const {
  // the cdf of a standard HalfCauchy is  F(x) = (2/pi) arctan(x)
  // therefore we will sample w ~ uniformly [0, pi/2] and compute tan(w)
  // finally we will multiply the scale to get the required value
  std::uniform_real_distribution<double> dist(0.0, M_PI_2);
  double value = in_nodes[0]->value._double * std::tan(dist(gen));
  return AtomicValue(AtomicType::POS_REAL, value);
}

// log_prob of a HalfCauchy is f(x; s) =  -log(pi/2) -log(s) -log(1 + (x/s)^2)
//   = -log(pi/2) + log(s) - log(s^2 + x^2)
// df/dx = -2x/(s^2 + x^2)
// d2f/dx2 = -2/(s^2 + x^2) + 4x^2/(s^2 + x^2)^2
// df/ds = 1/s -2s/(s^2 + x^2)
// d2f/ds2 = - 1/s^2 - 2/(s^2 + x^2) + 4s^2/(s^2 + x^2)^2

double HalfCauchy::log_prob(const AtomicValue& value) const {
  double x = value._double;
  double s = in_nodes[0]->value._double;
  return -std::log(M_PI_2) -std::log(s) - std::log1p(std::pow(x/s, 2));
}

void HalfCauchy::gradient_log_prob_value(
    const AtomicValue& value, double& grad1, double& grad2) const {
  double x = value._double;
  double s = in_nodes[0]->value._double;
  double s2_p_x2 = s * s + x * x;
  grad1 += - 2 * x / s2_p_x2;
  grad2 += - 2 / s2_p_x2 + 4 * x * x / (s2_p_x2 * s2_p_x2);
}

void HalfCauchy::gradient_log_prob_param(
    const AtomicValue& value, double& grad1, double& grad2) const {
  // gradients of s should be non-zero before computing gradients w.r.t. s
  double s_grad = in_nodes[0]->grad1;
  double s_grad2 = in_nodes[0]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double x = value._double;
    double s = in_nodes[0]->value._double;
    double s2_p_x2 = s * s + x * x;
    double grad_s = 1/s - 2 * s / s2_p_x2;
    double grad2_s2 = -1 / (s * s) - 2 / s2_p_x2 + 4 * s * s / (s2_p_x2 * s2_p_x2);
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

} // namespace distribution
} // namespace beanmachine
