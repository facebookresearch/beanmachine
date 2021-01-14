// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/linalgop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/unaryop.h"

namespace beanmachine {
namespace oper {

bool approx_zero(double val) {
  return std::abs(val) < graph::PRECISION;
}

// Note: that we use the following chain rule for the gradients of f(g(x))
// first: f'(g(x)) g'(x), assuming f'(g(x)) is given by back_grad1.

void UnaryOperator::backward() {
  assert(in_nodes.size() == 1);
  auto node = in_nodes[0];
  if (node->needs_gradient()) {
    node->back_grad1._double += back_grad1._double * jacobian();
  }
}

// g'(x) = -1
double Complement::jacobian() const {
  return -1.0;
}

// g'(x) = -1
double Negate::jacobian() const {
  return -1.0;
}

// g'(x) = exp(x) = g(x)
double Exp::jacobian() const {
  return value._double;
}

// g'(x) = exp(x) = g(x) + 1
double ExpM1::jacobian() const {
  return value._double + 1.0;
}

// 1st grad is the Normal(0, 1) pdf
// g'(x) = 1/sqrt(2 pi) exp(-0.5 x^2)
double Phi::jacobian() const {
  double x = in_nodes[0]->value._double;
  return M_SQRT1_2 * (M_2_SQRTPI / 2) * std::exp(-0.5 * x * x);
}

// g(x) = 1 / (1 + exp(-x))
// g'(x) = exp(-x) / (1 + exp(-x))^2 = g(x) * (1 - g(x))
double Logistic::jacobian() const {
  return value._double * (1 - value._double);
}

// g(x) = log (1 + exp(x))
// g'(x) = exp(x) / (1 + exp(x)) = 1 - exp(-g)
double Log1pExp::jacobian() const {
  return 1.0 - std::exp(-value._double);
}

// g(x) = log (1 - exp(x))
// g'(x) = -exp(x) / (1 - exp(x)) = 1 - exp(-g)
double Log1mExp::jacobian() const {
  return 1.0 - std::exp(-value._double);
}

// g'(x) = 1 / x
double Log::jacobian() const {
  return 1.0 / in_nodes[0]->value._double;
}

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

/*
For C = A @ B, with the backward accumulated gradient for C is Gc,
the backward propagation to A is Ga += Gc @ B^T and to B is
Gb += A^T @ Gc
*/
void MatrixMultiply::backward() {
  assert(in_nodes.size() == 2);
  auto node_a = in_nodes[0];
  auto node_b = in_nodes[1];
  Eigen::MatrixXd& A = node_a->value._matrix;
  Eigen::MatrixXd& B = node_b->value._matrix;
  // if C = A @ B is reduced to a scalar
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    if (node_a->needs_gradient()) {
      node_a->back_grad1._matrix += back_grad1._double * B.transpose();
    }
    if (node_b->needs_gradient()) {
      node_b->back_grad1._matrix += back_grad1._double * A.transpose();
    }
    return;
  }
  // the general form
  if (node_a->needs_gradient()) {
    node_a->back_grad1._matrix += back_grad1._matrix * B.transpose();
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1._matrix += A.transpose() * back_grad1._matrix;
  }
}

} // namespace oper
} // namespace beanmachine
