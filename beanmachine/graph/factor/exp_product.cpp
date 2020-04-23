// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/factor/exp_product.h"

namespace beanmachine {
namespace factor {

ExpProduct::ExpProduct(const std::vector<graph::Node*>& in_nodes)
    : Factor(graph::FactorType::EXP_PRODUCT) {
  // an ExpProduct factor must have at least one parent
  if (in_nodes.size() < 1) {
    throw std::invalid_argument(
        "ExpProduct factor needs at least one parent");
  }
  // the parent should be real, positive, or probability
  for (const graph::Node* parent: in_nodes) {
    if (parent->value.type != graph::AtomicType::REAL and
        parent->value.type != graph::AtomicType::POS_REAL and
        parent->value.type != graph::AtomicType::PROBABILITY) {
      throw std::invalid_argument("ExpProduct parents must be real, positive, or a probability");
    }
  }
}

// log_prob is actually quite simple for an exp_product of (x1, x2, .. xk)
// f(x1, x2, .. xk) = x1 * x2 *.. * xk
// df / dxi = x1 * x2 .. xi' .. xk
// d2f / d xi xj = 2 * x1 * .. xi' * .. xj' .. xk  for i != j
// d2f / d2 xi  = x1 * .. xi'' *  .. xk

double ExpProduct::log_prob() const {
  double product = 1.0;
  for (const Node* node: in_nodes) {
    product *= node->value._double;
  }
  return product;
}

void ExpProduct::gradient_log_prob(double& grad1, double& grad2) const {
  // we will use dynamic programming to compute the gradients in a single pass
  // product of previous terms
  double running_prod = 1;
  // product of previous terms and exactly one first gradient
  double running_prod_1grad = 0;
  // product of previous terms and exactly two gradients
  double running_prod_2grad = 0;
  for (const Node* node: in_nodes) {
    running_prod_2grad *= node->value._double;
    running_prod_2grad += 2 * running_prod_1grad * node->grad1 + running_prod * node->grad2;
    running_prod_1grad *= node->value._double;
    running_prod_1grad += running_prod * node->grad1;
    running_prod *= node->value._double;
  }
  grad1 += running_prod_1grad;
  grad2 += running_prod_2grad;
}

} // namespace factor
} // namespace beanmachine
