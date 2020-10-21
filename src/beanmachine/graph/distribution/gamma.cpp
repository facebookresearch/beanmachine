// Copyright 2004-present Facebook. All Rights Reserved.
#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/gamma.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Gamma::Gamma(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::GAMMA, sample_type) {
  // a Gamma distribution has two parents:
  // shape -> positive real; rate -> positive real
  if (sample_type != AtomicType::POS_REAL) {
    throw std::invalid_argument("Gamma produces positive real samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Gamma distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != AtomicType::POS_REAL or
      in_nodes[1]->value.type != AtomicType::POS_REAL) {
    throw std::invalid_argument("Gamma parents must be positive real-valued");
  }
}

double Gamma::_double_sampler(std::mt19937& gen) const {
  std::gamma_distribution<double> dist(
      in_nodes[0]->value._double, 1 / in_nodes[1]->value._double);
  return dist(gen);
}

// Note: log_prob(x | a, b) = a * log(b) - log G(a) + (a - 1) * log(x) - b * x
// grad1 w.r.t. x = (a - 1) / x - b
// grad2 w.r.t. x = (1 - a) / x^2
// grad1 w.r.t. params = [log(b) - digamma(a) + log(x)] * a'
//                     + (a / b - x) * b'
// grad2 w.r.t. params = - polygamma(1, a) * (a')^2
//                       + [log(b) - digamma(a) + log(x)] * a''
//                       - a / b^2 * (b')^2 + (a / b - x) * b''
double Gamma::log_prob(const graph::NodeValue& value) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double ret_val = param_a * std::log(param_b) - lgamma(param_a);
  ret_val +=
      (param_a - 1.0) * std::log(value._double) - param_b * value._double;
  return ret_val;
}

void Gamma::gradient_log_prob_value(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  grad1 += (param_a - 1.0) / value._double - param_b;
  grad2 += (1.0 - param_a) / (value._double * value._double);
}

void Gamma::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a = util::polygamma(0, param_a); // digamma(a)
  double poly1_a = util::polygamma(1, param_a); // polygamma(1, a)
  // 1st order derivatives
  double grad_a = std::log(param_b) - digamma_a + std::log(value._double);
  double grad_b = param_a / param_b - value._double;
  // 2nd order derivatives
  double grad2_a2 = -poly1_a;
  double grad2_b2 = -param_a / (param_b * param_b);
  // combine with chain rule
  grad1 += grad_a * in_nodes[0]->grad1 + grad_b * in_nodes[1]->grad1;
  grad2 += grad_a * in_nodes[0]->grad2 + grad_b * in_nodes[1]->grad2 +
      grad2_a2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      grad2_b2 * in_nodes[1]->grad1 * in_nodes[1]->grad1;
}

} // namespace distribution
} // namespace beanmachine
