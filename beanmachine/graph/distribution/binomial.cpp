// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/distribution/binomial.h"

namespace beanmachine {
namespace distribution {

Binomial::Binomial(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BINOMIAL, sample_type) {
  // a Binomial has two parents -- natural, probability and outputs a natural
  if (sample_type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument("Binomial produces natural number samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Binomial distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::NATURAL or
      in_nodes[1]->value.type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "Binomial parents must be a natural number and a probability");
  }
}

graph::AtomicValue Binomial::sample(std::mt19937& gen) const {
  graph::natural_t param_n = in_nodes[0]->value._natural;
  double param_p = in_nodes[1]->value._double;
  std::binomial_distribution<graph::natural_t> distrib(param_n, param_p);
  return graph::AtomicValue(distrib(gen));
}

double Binomial::log_prob(const graph::AtomicValue& value) const {
  graph::natural_t n = in_nodes[0]->value._natural;
  double p = in_nodes[1]->value._double;
  graph::natural_t k = value._natural;
  if (k > n) {
    return -std::numeric_limits<double>::infinity();
  }
  // we will try not to evaluate log(p) or log(1-p) unless needed
  double ret_val = 0;
  if (k > 0) {
    ret_val += k * log(p);
  }
  if (k < n) {
    ret_val += (n - k) * log(1 - p);
  }
  // note: Gamma(n+1) = n!
  ret_val += std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
  return ret_val;
}

// log_prob is k log(p) + (n-k) log(1-p) as a function of k
// grad1 is  (log(p) - log(1-p))
// grad2 is  0
void Binomial::gradient_log_prob_value(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  // nothing to do here since the value is a natural number and we can't
  // compute gradients w.r.t. naturals
  double p = in_nodes[1]->value._double;
  grad1 += std::log(p) - std::log(1 - p);
  // grad2 += 0;
}

// log_prob is k log(p) + (n-k) log(1-p) as a function of p
// grad1 is  (k/p) * p' - ((n-k) / (1-p)) * p'
// grad2 is -(k/p^2) * p'^2 + (k/p) * p'' - ((n-k)/(1-p)^2) * p'^2 -
// ((n-k)/(1-p)) * p''
void Binomial::gradient_log_prob_param(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  double n = (double)in_nodes[0]->value._natural;
  double p = in_nodes[1]->value._double;
  double k = (double)value._natural;
  // first compute gradients w.r.t. p
  double grad_p = (k / p) - (n - k) / (1 - p);
  double grad2_p2 = (-k / (p * p)) - (n - k) / ((1 - p) * (1 - p));
  grad1 += grad_p * in_nodes[1]->grad1;
  grad2 += grad2_p2 * in_nodes[1]->grad1 * in_nodes[1]->grad1 +
      grad_p * in_nodes[1]->grad2;
}

} // namespace distribution
} // namespace beanmachine
