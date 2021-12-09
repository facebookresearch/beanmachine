/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/factor/exp_product.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace factor {

ExpProduct::ExpProduct(const std::vector<graph::Node*>& in_nodes)
    : Factor(graph::FactorType::EXP_PRODUCT) {
  // an ExpProduct factor must have at least one parent
  if (in_nodes.size() < 1) {
    throw std::invalid_argument(
        "factor EXP_PRODUCT requires one or more parents");
  }

  // the parent should be real, positive, negative, or probability
  for (const graph::Node* parent : in_nodes) {
    if (parent->value.type != graph::AtomicType::REAL and
        parent->value.type != graph::AtomicType::POS_REAL and
        parent->value.type != graph::AtomicType::NEG_REAL and
        parent->value.type != graph::AtomicType::PROBABILITY) {
      throw std::invalid_argument(
          "factor EXP_PRODUCT requires real, pos_real, neg_real or probability parents");
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
  for (const Node* node : in_nodes) {
    product *= node->value._double;
  }
  return product;
}

void ExpProduct::gradient_log_prob(
    const graph::Node* target_node,
    double& grad1,
    double& grad2) const {
  // we will use dynamic programming to compute the gradients in a single pass
  // product of previous terms
  double running_prod = 1;
  // product of previous terms and exactly one first gradient
  double running_prod_1grad = 0;
  // product of previous terms and exactly two gradients
  double running_prod_2grad = 0;
  for (const Node* node : in_nodes) {
    running_prod_2grad *= node->value._double;
    running_prod_2grad +=
        2 * running_prod_1grad * node->grad1 + running_prod * node->grad2;
    running_prod_1grad *= node->value._double;
    running_prod_1grad += running_prod * node->grad1;
    running_prod *= node->value._double;
  }
  grad1 += running_prod_1grad;
  grad2 += running_prod_2grad;
}

// In backward mode, we add the dlog_prob(x1,...,xk)/dxi to each parent xi
// for ExpProduct, dlog_prob(x1,...,xk)/dxi = prod{xj} for all j!=i.
void ExpProduct::backward() {
  std::vector<graph::Node*> zeros;
  double non_zero_prod = 1.0;
  for (const auto node : in_nodes) {
    if (util::approx_zero(node->value._double)) {
      zeros.push_back(node);
    } else {
      non_zero_prod *= node->value._double;
    }
  }
  if (zeros.size() == 1 and zeros.front()->needs_gradient()) {
    // if there is only one zero, only its backgrad needs update
    zeros.front()->back_grad1._double += non_zero_prod;
    return;
  } else if (zeros.size() > 1) {
    // if multiple zeros, all grad increments are zero, no need to update
    return;
  }

  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1._double += non_zero_prod / node->value._double;
    }
  }
}

} // namespace factor
} // namespace beanmachine
