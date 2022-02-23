/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/linalgop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/unaryop.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace oper {

// This is a fairly technical code base because it requires
// familiarity with Automatic Differentiation (AD) and
// vector/matrix function AD. The following references help:
// For background on reverse mode differentiation, see:
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
// For matrix differentation, see:
// https://atmos.washington.edu/~dennis/MatrixCalculus.pdf
// or
// https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
// or
// https://researcher.watson.ibm.com/researcher/files/us-pederao/ADTalk.pdf
// Note: The reader will also need to understand the definition of
// differentiations on matrices to follow these references. For this the
// following reference helps:
// https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf

// Note: that we use the following chain rule for the gradients of f(g(x))
// first: f'(g(x)) g'(x), assuming f'(g(x)) is given by back_grad1.

// void UnaryOperator::backward() {
//   assert(in_nodes.size() == 1);
//   auto node = in_nodes[0];
//   if (node->needs_gradient()) {
//     node->back_grad1._value += back_grad1._value * jacobian();
//   }
// }

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
  return value._value.item().toDouble();
}

// g'(x) = exp(x) = g(x) + 1
double ExpM1::jacobian() const {
  return value._value.item().toDouble() + 1.0;
}

// 1st grad is the Normal(0, 1) pdf
// g'(x) = 1/sqrt(2 pi) exp(-0.5 x^2)
double Phi::jacobian() const {
  double x = in_nodes[0]->value._value;
  return M_SQRT1_2 * (M_2_SQRTPI / 2) * std::exp(-0.5 * x * x);
}

// g(x) = 1 / (1 + exp(-x))
// g'(x) = exp(-x) / (1 + exp(-x))^2 = g(x) * (1 - g(x))
double Logistic::jacobian() const {
  return (value._value * (1 - value._value)).item().toDouble();
}

// g(x) = log (1 + exp(x))
// g'(x) = exp(x) / (1 + exp(x)) = 1 - exp(-g)
double Log1pExp::jacobian() const {
  return 1.0 - (-value._value).exp().item().toDouble();
}

// g(x) = log (1 - exp(x))
// g'(x) = -exp(x) / (1 - exp(x)) = 1 - exp(-g)
double Log1mExp::jacobian() const {
  return 1.0 - (-value._value).exp().item().toDouble();
}

// g'(x) = 1 / x
double Log::jacobian() const {
  return 1.0 / in_nodes[0]->value._value.item().toDouble();
}

// dg(x1,...xn)/dxi = 1
void Add::backward() {
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1._value += back_grad1._value;
    }
  }
}

// dg(x1,...xn)/dxi = g(x1, ..., xn) / xi
void Multiply::backward() {
  // if at least one parent value is likely zero.
  if (util::approx_zero(value._value.item().toDouble())) {
    std::vector<graph::Node*> zeros;
    double non_zero_prod = 1.0;
    for (const auto node : in_nodes) {
      if (util::approx_zero(node->value._value.item().toDouble())) {
        zeros.push_back(node);
      } else {
        non_zero_prod *= node->value._value.item().toDouble();
      }
    }
    if (zeros.size() == 1) {
      // if there is only one zero, only its backgrad needs update
      zeros.front()->back_grad1._value += back_grad1._value * non_zero_prod;
      return;
    } else if (zeros.size() > 1) {
      // if multiple zeros, all grad increments are zero, no need to update
      return;
    } // otherwise, do the usual update
  }
  double shared_numerator = (back_grad1._value * value._value).item().toDouble();
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1._value += shared_numerator / node->value._value;
    }
  }
}

// g(x0, x1) = x0 ^ x1
// dg/x0 = x1 * x0 ^ (x1 - 1)
// dg/x1 = g * log(x0)
void Pow::backward() {
  assert(in_nodes.size() == 2);
  torch::Tensor x0 = in_nodes[0]->value._value;
  torch::Tensor x1 = in_nodes[1]->value._value;
  if (in_nodes[0]->needs_gradient()) {
    torch::Tensor jacob = util::approx_zero(x0.item().toDouble()) ? x1 * (x0).pow(x1 - 1)
                                         : value._value * x1 / x0;
    in_nodes[0]->back_grad1._value += back_grad1._value * jacob;
  }
  if (in_nodes[1]->needs_gradient()) {
    torch::Tensor jacob = value._value * x0.log();
    in_nodes[1]->back_grad1._value += back_grad1._value * jacob;
  }
}

// dg(x1,...xn)/dxi = exp(xi) / sum_j^n exp(xj) = exp(xi - g)
void LogSumExp::backward() {
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1._value +=
          back_grad1._value * (node->value._value - value._value).exp();
    }
  }
}

// dg(x1,...xn)/dxi = exp(xi) / sum_j^n exp(xj) = exp(xi - g)
void LogSumExpVector::backward() {
  if (in_nodes[0]->needs_gradient()) {
    torch::Tensor exp =
        (in_nodes[0]->value._value - value._value).exp();
    in_nodes[0]->back_grad1._value += back_grad1._value * exp;
  }
}

void IfThenElse::backward() {
  assert(in_nodes.size() == 3);
  int choice = in_nodes[0]->value._value.item().toBool() ? 1 : 2;
  if (in_nodes[choice]->needs_gradient()) {
    in_nodes[choice]->back_grad1._value += back_grad1._value;
  }
}

void Choice::backward() {
  graph::natural_t choice = in_nodes[0]->value._value.item().toInt() + 1;
  assert(in_nodes.size() < choice);
  if (in_nodes[choice]->needs_gradient()) {
    in_nodes[choice]->back_grad1._value += back_grad1._value;
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
  torch::Tensor& A = node_a->value._value;
  torch::Tensor& B = node_b->value._value;
  // if C = A @ B is reduced to a scalar
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    if (node_a->needs_gradient()) {
      node_a->back_grad1._value += back_grad1._value * B.transpose(0, 1);
    }
    if (node_b->needs_gradient()) {
      node_b->back_grad1._value += back_grad1._value * A.transpose(0, 1);
    }
    return;
  }
  // the general form
  if (node_a->needs_gradient()) {
    node_a->back_grad1._value += back_grad1._value * B.transpose(0, 1);
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1._value += A.transpose(0, 1) * back_grad1._value;
  }
}

// TODO[Walid]: The following needs to be modified to actually
// implement the desired functionality

/*
For C = A @ B, with the backward accumulated gradient for C is Gc,
the backward propagation to A is Ga += Gc @ B^T and to B is
Gb += A^T @ Gc
*/
void MatrixScale::backward() {
  assert(in_nodes.size() == 2);
  auto node_a = in_nodes[0];
  auto node_b = in_nodes[1];
  double A = node_a->value._value.item().toDouble();
  torch::Tensor& B = node_b->value._value;
  // if C = A @ B is reduced to a scalar
  // TODO[Walid] : Check if this case is actually ever needed
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    if (node_a->needs_gradient()) {
      node_a->back_grad1._value += (back_grad1._value * B.transpose(0, 1)).sum().item().toDouble();
    }
    if (node_b->needs_gradient()) {
      node_b->back_grad1._value[0][0] += back_grad1._value * A;
    }
    return;
  }
  // the general form
  if (node_a->needs_gradient()) {
    node_a->back_grad1._value += (back_grad1._value * B.transpose(0, 1)).sum().item().toDouble();
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1._value += A * back_grad1._value;
  }
}

void Index::backward() {
  assert(in_nodes.size() == 2);
  auto matrix = in_nodes[0];
  if (matrix->needs_gradient()) {
    matrix->back_grad1._value[in_nodes[1]->value._value] +=
        back_grad1._value;
  }
}

void ColumnIndex::backward() {
  assert(in_nodes.size() == 2);
  auto matrix = in_nodes[0];
  if (matrix->needs_gradient()) {
    matrix->back_grad1._value.index_put_(
      {torch::indexing::Slice(),(int)in_nodes[1]->value._value.item().toInt()},
      matrix->back_grad1._value.index(
        {torch::indexing::Slice(),(int)in_nodes[1]->value._value.item().toInt()}
      ) + back_grad1._value
    );
  }
}

void ToMatrix::backward() {
  int rows = static_cast<int>(in_nodes[0]->value._value.item().toInt());
  int cols = static_cast<int>(in_nodes[1]->value._value.item().toInt());
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      auto node = in_nodes[2 + j * rows + i];
      if (node->needs_gradient()) {
        node->back_grad1._value += back_grad1._value[i][j].item().toDouble();
      }
    }
  }
}

void BroadcastAdd::backward() {
  assert(in_nodes.size() == 2);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._value += back_grad1._value.sum().item().toDouble();
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._value += back_grad1._value;
  }
}

} // namespace oper
} // namespace beanmachine
