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

void UnaryOperator::backward() {
  assert(in_nodes.size() == 1);
  auto node = in_nodes[0];
  if (node->needs_gradient()) {
    node->back_grad1 += back_grad1 * jacobian();
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
      node->back_grad1 += back_grad1;
    }
  }
}

// dg(x1,...xn)/dxi = g(x1, ..., xn) / xi
void Multiply::backward() {
  // if at least one parent value is likely zero.
  if (util::approx_zero(value._double)) {
    std::vector<graph::Node*> zeros;
    double non_zero_prod = 1.0;
    for (const auto node : in_nodes) {
      if (util::approx_zero(node->value._double)) {
        zeros.push_back(node);
      } else {
        non_zero_prod *= node->value._double;
      }
    }
    if (zeros.size() == 1) {
      // if there is only one zero, only its backgrad needs update
      zeros.front()->back_grad1 += back_grad1 * non_zero_prod;
      return;
    } else if (zeros.size() > 1) {
      // if multiple zeros, all grad increments are zero, no need to update
      return;
    } // otherwise, do the usual update
  }
  double shared_numerator = back_grad1 * value._double;
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1 += shared_numerator / node->value._double;
    }
  }
}

// g(x0, x1) = x0 ^ x1
// dg/x0 = x1 * x0 ^ (x1 - 1)
// dg/x1 = g * log(x0)
void Pow::backward() {
  assert(in_nodes.size() == 2);
  double x0 = in_nodes[0]->value._double;
  double x1 = in_nodes[1]->value._double;
  if (in_nodes[0]->needs_gradient()) {
    double jacob = util::approx_zero(x0) ? x1 * std::pow(x0, x1 - 1)
                                         : value._double * x1 / x0;
    in_nodes[0]->back_grad1 += back_grad1 * jacob;
  }
  if (in_nodes[1]->needs_gradient()) {
    double jacob = value._double * std::log(x0);
    in_nodes[1]->back_grad1 += back_grad1 * jacob;
  }
}

// dg(x1,...xn)/dxi = exp(xi) / sum_j^n exp(xj) = exp(xi - g)
void LogSumExp::backward() {
  for (const auto node : in_nodes) {
    if (node->needs_gradient()) {
      node->back_grad1 +=
          back_grad1 * std::exp(node->value._double - value._double);
    }
  }
}

// dg(x1,...xn)/dxi = exp(xi) / sum_j^n exp(xj) = exp(xi - g)
void LogSumExpVector::backward() {
  if (in_nodes[0]->needs_gradient()) {
    Eigen::MatrixXd exp =
        (in_nodes[0]->value._matrix.array() - value._double).exp();
    in_nodes[0]->back_grad1 += graph::DoubleMatrix::times(back_grad1, exp);
  }
}

void IfThenElse::backward() {
  assert(in_nodes.size() == 3);
  int choice = in_nodes[0]->value._bool ? 1 : 2;
  if (in_nodes[choice]->needs_gradient()) {
    in_nodes[choice]->back_grad1 += back_grad1;
  }
}

void Choice::backward() {
  graph::natural_t choice = in_nodes[0]->value._natural + 1;
  assert(in_nodes.size() < choice);
  if (in_nodes[choice]->needs_gradient()) {
    in_nodes[choice]->back_grad1 += back_grad1;
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

  if (node_a->needs_gradient()) {
    node_a->back_grad1 += graph::DoubleMatrix::times(back_grad1, B.transpose());
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1 += graph::DoubleMatrix::times(A.transpose(), back_grad1);
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
  double A = node_a->value._double;
  Eigen::MatrixXd& B = node_b->value._matrix;
  // if C = A @ B is reduced to a scalar
  if (node_a->needs_gradient()) {
    node_a->back_grad1 +=
        graph::DoubleMatrix::times(back_grad1, B.transpose()).as_matrix().sum();
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1 += A * back_grad1;
  }
}

void ElementwiseMultiply::backward() {
  assert(in_nodes.size() == 2);
  auto node_a = in_nodes[0];
  auto node_b = in_nodes[1];
  Eigen::MatrixXd& A = node_a->value._matrix;
  Eigen::MatrixXd& B = node_b->value._matrix;

  if (node_a->needs_gradient()) {
    node_a->back_grad1 += (back_grad1.array() * B.array()).matrix();
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1 += (back_grad1.array() * A.array()).matrix();
  }
}

void MatrixAdd::backward() {
  assert(in_nodes.size() == 2);
  auto node_a = in_nodes[0];
  auto node_b = in_nodes[1];
  if (node_a->needs_gradient()) {
    node_a->back_grad1 += back_grad1;
  }
  if (node_b->needs_gradient()) {
    node_b->back_grad1 += back_grad1;
  }
}

void Index::backward() {
  assert(in_nodes.size() == 2);
  auto matrix = in_nodes[0];
  if (matrix->needs_gradient()) {
    matrix->back_grad1(in_nodes[1]->value._natural) += back_grad1;
  }
}

void ColumnIndex::backward() {
  assert(in_nodes.size() == 2);
  auto matrix = in_nodes[0];
  if (matrix->needs_gradient()) {
    // TODO: not used in any tests; need to create some.
    matrix->back_grad1.col(in_nodes[1]->value._natural) += back_grad1;
  }
}

void ToMatrix::backward() {
  int rows = static_cast<int>(in_nodes[0]->value._natural);
  int cols = static_cast<int>(in_nodes[1]->value._natural);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      auto node = in_nodes[2 + j * rows + i];
      if (node->needs_gradient()) {
        node->back_grad1 += back_grad1(i, j);
      }
    }
  }
}

void BroadcastAdd::backward() {
  assert(in_nodes.size() == 2);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 += back_grad1.sum();
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1 += back_grad1;
  }
}

void Cholesky::backward() {
  assert(in_nodes.size() == 1);
  // We compute the gradient in place on a copy of the upstream gradient
  // according to the algorithm described in section 4.1 of
  // https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
  if (in_nodes[0]->needs_gradient()) {
    uint n = in_nodes[0]->value.type.rows;
    Eigen::MatrixXd L = value._matrix;
    Eigen::MatrixXd dS = back_grad1.as_matrix().triangularView<Eigen::Lower>();
    for (int i = n - 1; i >= 0; i--) {
      // update grad dS at lower-triangular col i, including (i,i)
      Eigen::VectorXd L_c = L(Eigen::seq(i + 1, Eigen::last), i);
      Eigen::VectorXd dS_c = dS(Eigen::seq(i + 1, Eigen::last), i);
      dS(i, i) -= L_c.dot(dS_c) / L(i, i);
      dS(Eigen::seq(i, Eigen::last), i) /= L(i, i);

      if (i > 0) {
        // update grad dS at lower-triangular row i (excluding i,i)
        Eigen::MatrixXd L_r = L(i, Eigen::seq(0, i - 1));
        Eigen::MatrixXd L_rB =
            L(Eigen::seq(i, Eigen::last), Eigen::seq(0, i - 1));

        dS(i, Eigen::seq(0, i - 1)) -=
            dS(Eigen::seq(i, Eigen::last), i).transpose() * L_rB;

        // update grad dS at lower-triangular block left/below index (i, i)
        dS(Eigen::seq(i + 1, Eigen::last), Eigen::seq(0, i - 1)) -=
            dS(Eigen::seq(i + 1, Eigen::last), i) * L_r;
      }

      dS(i, i) /= 2;
    }

    // split gradient between upper and lower triangular parts of input,
    // which are symmetric. This follows the convention used by Pytorch,
    // while the Iain Murray description accumulates all gradients
    // to the lower triangular part.
    for (uint i = 0; i < n; i++) {
      for (uint j = i + 1; j < n; j++) {
        dS(j, i) /= 2;
        dS(i, j) = dS(j, i);
      }
    }

    in_nodes[0]->back_grad1 += dS;
  }
}

void MatrixExp::backward() {
  assert(in_nodes.size() == 1);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 +=
        back_grad1.as_matrix().cwiseProduct(value._matrix);
  }
}

void MatrixLog::backward() {
  assert(in_nodes.size() == 1);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 +=
        back_grad1.as_matrix().cwiseQuotient(in_nodes[0]->value._matrix);
  }
}

void MatrixSum::backward() {
  assert(in_nodes.size() == 1);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 = in_nodes[0]->back_grad1.array() + back_grad1;
  }
}

void Log1p::backward() {
  assert(in_nodes.size() == 1);
  if (in_nodes[0]->needs_gradient()) {
    double jacob = 1 / (1 + in_nodes[0]->value._double);
    in_nodes[0]->back_grad1 += back_grad1 * jacob;
  }
}

void MatrixLog1p::backward() {
  assert(in_nodes.size() == 1);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 += back_grad1.as_matrix().cwiseQuotient(
        (in_nodes[0]->value._matrix.array() + 1).matrix());
  }
}

} // namespace oper
} // namespace beanmachine
