// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

Beta::Beta(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BETA, sample_type) {
  // a Beta has two parents which are real numbers and it outputs a probability
  if (sample_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Beta produces probability samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Beta distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL
      or in_nodes[1]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("Beta parents must be positive real-valued");
  }
}

graph::AtomicValue Beta::sample(std::mt19937& gen) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  return graph::AtomicValue(
    graph::AtomicType::PROBABILITY, util::sample_beta(gen, param_a, param_b));
}

double Beta::log_prob(const graph::AtomicValue& value) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double ret_val = (param_a - 1) * log(value._double)
    + (param_b - 1) * log(1 - value._double);
  ret_val += lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);
  return ret_val;
}

// Note log_prob(x | a, b) = (a-1) log(x) + (b-1) log(1-x) + log G(a+b) - log G(a) - log G(b)
// grad1 w.r.t. x =  (a-1) / x - (b-1) / (1-x)
// grad2 w.r.t. x = - (a-1) / x^2 - (b-1) / (1-x)^2
// grad1 w.r.t. params = (log(x) + digamma(a+b) - digamma(a)) a'
//                        + (log(1-x) + digamma(a+b) - digamma(b)) b'
// grad2 w.r.t. params =
//   (polygamma(1, a+b) - polygamma(1, a)) a'^2 + (log(x) + digamma(a+b) - digamma(a)) a''
//   (polygamma(1, a+b) - polygamma(1, b)) b'^2 + (log(1-x) + digamma(a+b) - digamma(b)) b''

void Beta::gradient_log_prob_value(
    const graph::AtomicValue& value, double& grad1, double& grad2) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  grad1 += (param_a - 1) / value._double - (param_b - 1) / (1 - value._double);
  grad2 += - (param_a - 1) / (value._double * value._double)
    - (param_b - 1) / ((1 - value._double) * (1 - value._double));
}

void Beta::gradient_log_prob_param(
    const graph::AtomicValue& value, double& grad1, double& grad2) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a_p_b = util::polygamma(0, param_a + param_b); // digamma(a+b)
  double poly1_a_p_b = util::polygamma(1, param_a + param_b); // polygamma(1, a+b)
  // first compute gradients w.r.t. a and b
  double grad_a = std::log(value._double) + digamma_a_p_b - util::polygamma(0, param_a);
  double grad_b = std::log(1 - value._double) + digamma_a_p_b - util::polygamma(0, param_b);
  double grad2_a2 = poly1_a_p_b - util::polygamma(1, param_a);
  double grad2_b2 = poly1_a_p_b - util::polygamma(1, param_b);
  grad1 += grad_a * in_nodes[0]->grad1 + grad_b * in_nodes[1]->grad1;
  grad2 +=  grad_a * in_nodes[0]->grad2 + grad_b * in_nodes[1]->grad2
    + grad2_a2 * in_nodes[0]->grad1 * in_nodes[0]->grad1
    + grad2_b2 * in_nodes[1]->grad1 * in_nodes[1]->grad1;
}

} // namespace distribution
} // namespace beanmachine
