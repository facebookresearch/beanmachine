// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/bernoulli_noisy_or.h"

namespace beanmachine {
namespace distribution {

// helper function to compute log(1 - exp(-x)) for x >= 0
static inline double log1mexpm(double x) {
  // see the following document for an explanation of why we switch at .69315
  // https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  // essentially for small values expm1 prevents underflow to 0.0 and for
  // large values it prevents overflow to 1.0.
  //   log1mexp(1e-20) -> -46.051701859880914
  //   log1mexp(40) -> -4.248354255291589e-18
  if (x < 0.69315) {
    return std::log(-std::expm1(-x));
  }
  else {
    return std::log1p(-std::exp(-x));
  }
}

BernoulliNoisyOr::BernoulliNoisyOr(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BERNOULLI, sample_type) {
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "BernoulliNoisyOr produces boolean valued samples");
  }
  // a BernoulliNoisyOr can only have one parent which must be a POS_REAL (>=0)
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "BernoulliNoisyOr distribution must have exactly one parent");
  }
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "BernoulliNoisyOr parent probability must be positive real-valued");
  }
}

graph::AtomicValue BernoulliNoisyOr::sample(std::mt19937& gen) const {
  double param = in_nodes[0]->value._double;
  double prob = 1 - exp(-param);
  std::bernoulli_distribution distrib(prob);
  return graph::AtomicValue((bool)distrib(gen));
}

double BernoulliNoisyOr::log_prob(const graph::AtomicValue& value) const {
  double param = in_nodes[0]->value._double;
  if (not value._bool) {
    return -param;
  } else {
    return log1mexpm(param);
  }
}

// x ~ BernoulliNoisyOr(y) = Bernoulli(1 - exp(-y))
// f(x, y) = x log(1 - exp(-y)) + (1-x)(-y)
// w.r.t. x:   f' = log(1 - exp(-y)) + y     f'' = 0
// w.r.t. y:   f' =  [x exp(-y) / (1 - exp(-y)) - (1-x)] y' = [x / (1 - exp(-y)) - 1] y'
//             f'' = [- x exp(-y) /(1 - exp(-y))^2] y'^2 + [x / (1 - exp(-y)) - 1] y''
void BernoulliNoisyOr::gradient_log_prob_value(
    const graph::AtomicValue& value, double& grad1, double& grad2) const {
  double param = in_nodes[0]->value._double;
  grad1 += log1mexpm(param) + param;
  // grad2 += 0
}

void BernoulliNoisyOr::gradient_log_prob_param(
    const graph::AtomicValue& value, double& grad1, double& grad2) const {
  double param = in_nodes[0]->value._double;
  double mexpm1m = -std::expm1(-param); // 1 - exp(-param)
  double val = (double) value._bool;
  double grad_param= val / mexpm1m - 1;
  double grad2_param = - val * std::exp(-param) / (mexpm1m * mexpm1m);
  grad1 += grad_param * in_nodes[0]->grad1;
  grad2 += grad2_param * in_nodes[0]->grad1 * in_nodes[0]->grad1
    + grad_param * in_nodes[0]->grad2;
}

} // namespace distribution
} // namespace beanmachine
