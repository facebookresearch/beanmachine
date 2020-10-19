// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/bernoulli_logit.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

BernoulliLogit::BernoulliLogit(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::BERNOULLI_LOGIT, sample_type) {
  // a BernoulliLogit distribution has one parent which is a real value
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "BernoulliLogit distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::REAL) {
    throw std::invalid_argument("BernoulliLogit parent must be a real value");
  }
  // only BOOLEAN-valued samples are possible
  if (sample_type != AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "BernoulliLogit distribution produces boolean samples");
  }
}

bool BernoulliLogit::_bool_sampler(std::mt19937& gen) const {
  double logodds = in_nodes[0]->value._double;
  return (bool)util::sample_logodds(gen, logodds);
}

// log_prob of a BernoulliLogit with parameter l (logodds)
// f(x, l) = - x log(1 + exp(-l)) - (1-x) log(1 + exp(l))
// df / dx = - log(1 + exp(-l)) + log(1 + exp(l))
//         = l
// d2f / dx2 = 0
// df / dl = x exp(-l) / (1 + exp(-l)) - (1-x) exp(l) / (1 + exp(l))
//          = x /(1 + exp(l)) - (1 - x) / (1 + exp(-l))
// d2f dl2 = - x exp(l) / (1 + exp(l))^2 - (1-x) exp(-l) / (1 + exp(-l))^2
//         = -1 /[(1 + exp(-l)) (1 + exp(l))]
//         = -1 / (2 + exp(-l) + exp(l))

double BernoulliLogit::log_prob(const NodeValue& value) const {
  bool x = value._bool;
  double l = in_nodes[0]->value._double;
  return x ? -util::log1pexp(-l) : -util::log1pexp(l);
}

void BernoulliLogit::gradient_log_prob_value(
    const NodeValue& /* value */,
    double& grad1,
    double& /* grad2 */) const {
  double l = in_nodes[0]->value._double;
  grad1 += l;
  // grad2 += 0
}

void BernoulliLogit::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  bool x = value._bool;
  double l = in_nodes[0]->value._double;
  // We will compute the gradients w.r.t. each the parameter only if
  // the gradients of the parameter w.r.t. the source variable is non-zero
  double l_grad = in_nodes[0]->grad1;
  double l_grad2 = in_nodes[0]->grad2;
  if (l_grad != 0 or l_grad2 != 0) {
    double grad_l = x ? 1 / (1 + std::exp(l)) : -1 / (1 + std::exp(-l));
    double grad2_l2 = -1 / (2 + std::exp(-l) + std::exp(l));
    grad1 += grad_l * l_grad;
    grad2 += grad2_l2 * l_grad * l_grad + grad_l * l_grad2;
  }
}

} // namespace distribution
} // namespace beanmachine
