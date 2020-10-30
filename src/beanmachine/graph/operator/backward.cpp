// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/unaryop.h"

namespace beanmachine {
namespace oper {

bool approx_zero(double val) {
  return std::abs(val) < graph::PRECISION;
}

// Note: that we use the following chain rule for the gradients of f(g(x))
// first: f'(g(x)) g'(x), assuming f'(g(x)) is given by back_grad1.

// dg(x1,...xn)/dxi = 1
void Add::backward() {
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1._double += back_grad1._double;
    }
  }
}

// dg(x1,...xn)/dxi = g(x1, ..., xn) / xi
void Multiply::backward() {
  // if at least one parent value is likely zero.
  if (approx_zero(value._double)) {
    std::vector<graph::Node*> zeros;
    double non_zero_prod = 1.0;
    for (const auto node : in_nodes) {
      if (approx_zero(node->value._double)) {
        zeros.push_back(node);
      } else {
        non_zero_prod *= node->value._double;
      }
    }
    if (zeros.size() == 1) {
      // if there is only one zero, only its backgrad needs update
      zeros.front()->back_grad1._double += back_grad1._double * non_zero_prod;
      return;
    } else if (zeros.size() > 1) {
      // if multiple zeros, all grad increments are zero, no need to update
      return;
    } // otherwise, do the usual update
  }
  double shared_numerator = back_grad1._double * value._double;
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1._double += shared_numerator / node->value._double;
    }
  }
}

} // namespace oper
} // namespace beanmachine
