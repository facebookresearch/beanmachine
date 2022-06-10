/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/linalgop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/unaryop.h"

using namespace beanmachine::distribution;

namespace beanmachine {
namespace oper {

// Note: that we use the following chain rule for the gradients of f(g(x))
// first: f'(g(x)) g'(x)
// second: f''(g(x)) g'(x)^2 + f'(g(x))g''(x)

void Complement::compute_gradients() {
  assert(in_nodes.size() == 1);
  // for complement (f(y)=1-y) and negate(f(y)=-y): f'(y) = -1 and f''(y) = 0
  grad1 = -1 * in_nodes[0]->grad1;
  grad2 = -1 * in_nodes[0]->grad2;
}

void ToInt::compute_gradients() {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void ToReal::compute_gradients() {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void ToRealMatrix::compute_gradients() {
  assert(in_nodes.size() == 1);
  Grad1 = in_nodes[0]->Grad1;
  Grad2 = in_nodes[0]->Grad2;
}

void ToPosReal::compute_gradients() {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void ToPosRealMatrix::compute_gradients() {
  assert(in_nodes.size() == 1);
  Grad1 = in_nodes[0]->Grad1;
  Grad2 = in_nodes[0]->Grad2;
}

void ToProbability::compute_gradients() {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void ToNegReal::compute_gradients() {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void Negate::compute_gradients() {
  assert(in_nodes.size() == 1);
  grad1 = -1 * in_nodes[0]->grad1;
  grad2 = -1 * in_nodes[0]->grad2;
}

void Exp::compute_gradients() {
  assert(in_nodes.size() == 1);
  // for f(y) = exp(y) or f(y) = exp(y)-1 we have f'(y) = exp(y) and f''(y) =
  // exp(y)
  double exp_parent = std::exp(in_nodes[0]->value._double);
  grad1 = exp_parent * in_nodes[0]->grad1;
  grad2 = grad1 * in_nodes[0]->grad1 + exp_parent * in_nodes[0]->grad2;
}

void ExpM1::compute_gradients() {
  assert(in_nodes.size() == 1);
  double exp_parent = std::exp(in_nodes[0]->value._double);
  grad1 = exp_parent * in_nodes[0]->grad1;
  grad2 = grad1 * in_nodes[0]->grad1 + exp_parent * in_nodes[0]->grad2;
}

void Log1pExp::compute_gradients() {
  assert(in_nodes.size() == 1);
  // f(x) = log (1 + exp(x))
  // f'(x) = exp(x) / (1 + exp(x)) = 1 - exp(-f)
  // f''(x) = exp(x) / (1 + exp(x))^2 = f' * (1 - f')
  double f_x = value._double;
  double f_grad = 1.0 - std::exp(-f_x);
  double f_grad2 = f_grad * (1.0 - f_grad);
  grad1 = f_grad * in_nodes[0]->grad1;
  grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      f_grad * in_nodes[0]->grad2;
}

void Log1mExp::compute_gradients() {
  assert(in_nodes.size() == 1);
  // f(x) = log (1 - exp(x))
  // f'(x) = -exp(x) / (1 - exp(x)) = 1 - exp(-f)
  // f''(x) = -exp(x) / (1 - exp(x))^2 = f' * (1 - f')
  double f_x = value._double;
  double f_grad = 1.0 - std::exp(-f_x);
  double f_grad2 = f_grad * (1.0 - f_grad);
  grad1 = f_grad * in_nodes[0]->grad1;
  grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      f_grad * in_nodes[0]->grad2;
}

void Log::compute_gradients() {
  assert(in_nodes.size() == 1);
  // f(x) = log(x)
  // f'(x) = 1 / x
  // f''(x) = -1 / (x^2) = -f'(x) * f'(x)
  double x = in_nodes[0]->value._double;
  double f_grad = 1.0 / x;
  double f_grad2 = -f_grad * f_grad;
  grad1 = f_grad * in_nodes[0]->grad1;
  grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      f_grad * in_nodes[0]->grad2;
}

void Phi::compute_gradients() {
  assert(in_nodes.size() == 1);
  // gradient of the cumulative of the normal density is simply
  // the normal density pdf:
  // 1/sqrt(2 pi) exp(-0.5 x^2)
  // second gradient = 1/sqrt(2 pi) exp(-0.5 x^2) * (-x)
  double x = in_nodes[0]->value._double;
  // compute gradient w.r.t. x and then include the gradient of x w.r.t. src
  // idx
  double grad1_x = M_SQRT1_2 * (M_2_SQRTPI / 2) * std::exp(-0.5 * x * x);
  double grad2_x = grad1_x * (-x);
  grad1 = grad1_x * in_nodes[0]->grad1;
  grad2 = grad2_x * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      grad1_x * in_nodes[0]->grad2;
}

void Logistic::compute_gradients() {
  assert(in_nodes.size() == 1);
  // f(x) = 1 / (1 + exp(-x))
  // f'(x) = exp(-x) / (1 + exp(-x))^2 = f(x) * (1 - f(x))
  // f''(x) = f'(x) - 2 f(x) f'(x) = f'(x) (1 - 2 f(x))
  double f_x = value._double;
  double f_grad = f_x * (1 - f_x);
  double f_grad2 = f_grad * (1 - 2 * f_x);
  grad1 = f_grad * in_nodes[0]->grad1;
  grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      f_grad * in_nodes[0]->grad2;
}

void Pow::compute_gradients() {
  assert(in_nodes.size() == 2);
  // We wish to compute the first and second derivatives of x ** y.
  // Note that the derivatives here are with respect to some implicit
  // variable (say, z), of which x and y are functions x(z) and y(z) of.
  //
  // If y is a constant and does not depend on z, then the computation
  // is straightforward and uses the power rule and product rule
  // (x(z) ** y)' = y * (x ** (y - 1)) * x'
  // (x(z) ** y)'' = y * (y - 1) * (x ** (y - 2)) * (x' ** 2) +
  //                 y * (x ** (y - 1)) * x''
  //
  // However, for y(z) that does depend on z,
  // x(z) ** y(z) is not a function we have a ready-made derivative formula for.
  // However we can get a more convenient form:
  // x ** y = exp(log (x ** y)) = exp(y log x)
  // Now we have a composition of exp and product, so we can write:
  // (x ** y)' = exp(y log x)*(y log x)'
  //
  // Note that exp(y log x) is x ** y itself,
  // so (x ** y)' = (x ** y)*(y log x)'
  // For short, let us define f and g as follows:
  // f = x ** y
  // g = y log x
  //
  // Then from the above we have
  // f' = f g'
  // f'' = f' g' + f g''
  // So if we compute g' and g'', we are done.
  //
  // g' = (y log x)' = y' log x + y 1/x x'
  // Let's call the two terms of g' m and n:
  // m = y' log x
  // n = y 1/x x' = x' y / x
  // Then,
  // g'  = m + n
  // g'' = m' + n'
  // m'  = y'' log x + x' y' / x
  // n'  = (x'' y / x  +  x' y' / x  -  x' x' y) / (x x)
  //
  // Note that m' and n' have a term in common, c = x' y' / x
  // so we can write:
  // m'  = y'' log x + c
  // n'  = (x'' y / x  + c  - x' x' y) / (x x)

  double f = value._double;
  double x = in_nodes[0]->value._double;
  double y = in_nodes[1]->value._double;
  double x1 = in_nodes[0]->grad1;
  double y1 = in_nodes[1]->grad1;
  double x2 = in_nodes[0]->grad2;
  double y2 = in_nodes[1]->grad2;

  // check if the exponent is a constant (has no parents)
  bool constant_exponent = in_nodes[1]->in_nodes.size() == 0;
  if (constant_exponent) {
    // (x^c)' = c * x^(c-1) * x'
    // (x^c)'' = c * (c-1) * x^(c-2) * x'^2 + c * x^(c-1) * x''
    grad1 = y * std::pow(x, y - 1) * x1;
    grad2 = y * (y - 1) * std::pow(x, y - 2) * x1 * x1 +
        y * std::pow(x, y - 1) * x2;
  } else {
    if (x <= 0) {
      // if x <= 0, then the gradient should be NaN
      grad1 = std::nan("");
      grad2 = std::nan("");
    } else {
      // use computation described above
      double logx = std::log(x);
      double m = y1 * logx;
      double n = x1 * y / x;
      double g1 = m + n;
      double f1 = g1 * f;
      double c = x1 * y1 / x;
      double m1 = y2 * logx + c;
      double n1 = x2 * y / x + c - x1 * x1 * y / (x * x);
      double g2 = m1 + n1;
      double f2 = g2 * f + g1 * f1;
      grad1 = f1;
      grad2 = f2;
    }
  }
}

void Add::compute_gradients() {
  grad1 = grad2 = 0;
  for (const auto node : in_nodes) {
    grad1 += node->grad1;
    grad2 += node->grad2;
  }
}

void MatrixMultiply::compute_gradients() {
  assert(in_nodes.size() == 2);
  int rows = static_cast<int>(in_nodes[0]->value.type.rows);
  int cols = static_cast<int>(in_nodes[1]->value.type.cols);
  Grad1 = Eigen::MatrixXd::Zero(rows, cols);
  Grad2 = Eigen::MatrixXd::Zero(rows, cols);

  bool parent_0_has_grad = in_nodes[0]->Grad1.size() != 0;
  bool parent_1_has_grad = in_nodes[1]->Grad1.size() != 0;
  if (parent_0_has_grad) {
    Grad1 += in_nodes[0]->Grad1 * in_nodes[1]->value._matrix;
    Grad2 += in_nodes[0]->Grad2 * in_nodes[1]->value._matrix;
  }
  if (parent_1_has_grad) {
    Grad1 += in_nodes[0]->value._matrix * in_nodes[1]->Grad1;
    Grad2 += in_nodes[0]->value._matrix * in_nodes[1]->Grad2;
  }
  if (parent_0_has_grad and parent_1_has_grad) {
    Grad2 += 2 * (in_nodes[0]->Grad1 * in_nodes[1]->Grad1);
  }
}

void Multiply::compute_gradients() {
  // in general, computing the first and second derivatives of a product
  // would have a quadratic number of terms to add if we naively applied
  // the chain rule. Here we are doing this in linear time using
  // dynamic programming.
  // at each point in the loop we will keep the following running terms
  double product = 1.0; // the product so far
  // the sum of all products with exactly one variable replaced
  // with its first grad
  double sum_product_one_grad1 = 0.0;
  // the sum of all products with exactly two distinct variables
  // replaced with their second grad
  double sum_product_two_grad1 = 0.0;
  // the sum of all products with exactly one variable replaced with its
  // second grad
  double sum_product_one_grad2 = 0.0;
  for (const auto in_node : in_nodes) {
    sum_product_one_grad2 *= in_node->value._double;
    sum_product_one_grad2 += product * in_node->grad2;
    sum_product_two_grad1 *= in_node->value._double;
    sum_product_two_grad1 += sum_product_one_grad1 * in_node->grad1;
    sum_product_one_grad1 *= in_node->value._double;
    sum_product_one_grad1 += product * in_node->grad1;
    product *= in_node->value._double;
  }
  grad1 = sum_product_one_grad1;
  grad2 = sum_product_two_grad1 * 2 + sum_product_one_grad2;
}

void ElementwiseMultiply::compute_gradients() {
  assert(in_nodes.size() == 2);
  int rows = static_cast<int>(in_nodes[1]->value.type.rows);
  int cols = static_cast<int>(in_nodes[1]->value.type.cols);
  Grad1 = Eigen::MatrixXd::Zero(rows, cols);
  Grad2 = Eigen::MatrixXd::Zero(rows, cols);

  bool parent_0_has_grad1 = in_nodes[0]->Grad1.size() != 0;
  bool parent_1_has_grad1 = in_nodes[1]->Grad1.size() != 0;
  bool parent_0_has_grad2 = in_nodes[0]->Grad2.size() != 0;
  bool parent_1_has_grad2 = in_nodes[1]->Grad2.size() != 0;
  if (parent_0_has_grad1) {
    Grad1 += (in_nodes[0]->Grad1.array() * in_nodes[1]->value._matrix.array())
                 .matrix();
  }
  if (parent_1_has_grad1) {
    Grad1 += (in_nodes[1]->Grad1.array() * in_nodes[0]->value._matrix.array())
                 .matrix();
  }
  if (parent_0_has_grad2) {
    Grad2 += (in_nodes[0]->Grad2.array() * in_nodes[1]->value._matrix.array())
                 .matrix();
  }
  if (parent_1_has_grad2) {
    Grad2 += (in_nodes[1]->Grad2.array() * in_nodes[0]->value._matrix.array())
                 .matrix();
  }
  if (parent_0_has_grad1 and parent_1_has_grad1) {
    Grad2 +=
        2 * (in_nodes[0]->Grad1.array() * in_nodes[1]->Grad1.array()).matrix();
  }
}

void LogSumExp::compute_gradients() {
  // f(g1, ..., gn) = log(sum_i^n exp(gi))
  // note: in the following equations, df/dx means partial derivative
  // grad1 = df/dx = sum_i^n (df/dgi * dgi/dx)
  // where df/dgi = exp(gi) / sum_j^n exp(gj) = exp(gi - f)
  // grad2 = d(df/dx)/dx =
  // sum_i^n{ d(df/dgi)/dx * dgi/dx + df/dgi * d(dgi/dx)/dx }
  // where d(df/dgi)/dx = exp(gi - f) * (dgi/dx - df/dx)
  // therefore, grad2 = sum_i^n{exp(gi - f) * [(dgi/dx - grad1)*dgi/dx +
  // d(dgi/dx)/dx]}
  grad1 = grad2 = 0;
  const uint N = static_cast<uint>(in_nodes.size());
  std::vector<double> f_grad;
  for (uint i = 0; i < N; i++) {
    const auto node_i = in_nodes[i];
    double f_grad_i = std::exp(node_i->value._double - value._double);
    grad1 += f_grad_i * node_i->grad1;
    f_grad.push_back(f_grad_i);
  }
  assert(f_grad.size() == N);
  assert(f_grad.size() > 0);
  for (uint i = 0; i < N; i++) {
    const auto node_i = in_nodes[i];
    // @lint-ignore (HOWTOEVEN) LocalUncheckedArrayBounds
    grad2 +=
        f_grad[i] * (node_i->grad1 * (node_i->grad1 - grad1) + node_i->grad2);
  }
}

void LogSumExpVector::compute_gradients() {
  // f(g1, ..., gn) = log(sum_i^n exp(gi))
  // note: in the following equations, df/dx means partial derivative
  // grad1 = df/dx = sum_i^n (df/dgi * dgi/dx)
  // where df/dgi = exp(gi) / sum_j^n exp(gj) = exp(gi - f)
  // grad2 = d(df/dx)/dx =
  // sum_i^n{ d(df/dgi)/dx * dgi/dx + df/dgi * d(dgi/dx)/dx }
  // where d(df/dgi)/dx = exp(gi - f) * (dgi/dx - df/dx)
  // therefore, grad2 = sum_i^n{exp(gi - f) * [(dgi/dx - grad1)*dgi/dx +
  // d(dgi/dx)/dx]}
  Eigen::MatrixXd f_grad =
      (in_nodes[0]->value._matrix.array() - value._double).exp();
  grad1 = (f_grad.array() * in_nodes[0]->Grad1.array()).sum();
  grad2 = (f_grad.array() *
           (in_nodes[0]->Grad1.array() * (in_nodes[0]->Grad1.array() - grad1) +
            in_nodes[0]->Grad2.array()))
              .sum();
}

void IfThenElse::compute_gradients() {
  assert(in_nodes.size() == 3);
  int choice = in_nodes[0]->value._bool ? 1 : 2;
  grad1 = in_nodes[choice]->grad1;
  grad2 = in_nodes[choice]->grad2;
}

void Choice::compute_gradients() {
  graph::natural_t choice = in_nodes[0]->value._natural + 1;
  assert(in_nodes.size() < choice);
  grad1 = in_nodes[choice]->grad1;
  grad2 = in_nodes[choice]->grad2;
}

void Index::compute_gradients() {
  assert(in_nodes.size() == 2);
  grad1 = in_nodes[0]->Grad1.coeff(in_nodes[1]->value._natural);
  grad2 = in_nodes[0]->Grad2.coeff(in_nodes[1]->value._natural);
}

void ColumnIndex::compute_gradients() {
  assert(in_nodes.size() == 2);
  int rows = static_cast<int>(in_nodes[0]->Grad1.rows());
  Grad1.resize(rows, 1);
  Grad2.resize(rows, 1);
  Grad1 = in_nodes[0]->Grad1.col(in_nodes[1]->value._natural);
  Grad2 = in_nodes[0]->Grad2.col(in_nodes[1]->value._natural);
}

void ToMatrix::compute_gradients() {
  int rows = static_cast<int>(in_nodes[0]->value._natural);
  int cols = static_cast<int>(in_nodes[1]->value._natural);
  Grad1.resize(rows, cols);
  Grad2.resize(rows, cols);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      Grad1(i, j) = in_nodes[2 + j * rows + i]->grad1;
      Grad2(i, j) = in_nodes[2 + j * rows + i]->grad2;
    }
  }
}

void BroadcastAdd::compute_gradients() {
  assert(in_nodes.size() == 2);
  auto rows = in_nodes[1]->value.type.rows;
  auto cols = in_nodes[1]->value.type.cols;
  Grad1.setConstant(rows, cols, in_nodes[0]->grad1);
  Grad2.setConstant(rows, cols, in_nodes[0]->grad2);

  // in_nodes[1] can be a constant matrix that does not have Grad1/Grad2
  // initialized
  if (in_nodes[1]->Grad1.size() != 0) {
    Grad1 += in_nodes[1]->Grad1;
  }
  if (in_nodes[1]->Grad2.size() != 0) {
    Grad2 += in_nodes[1]->Grad2;
  }
}

void MatrixAdd::compute_gradients() {
  auto rows = in_nodes[0]->value.type.rows;
  auto cols = in_nodes[0]->value.type.cols;
  Grad1.setZero(rows, cols);
  Grad2.setZero(rows, cols);
  if (in_nodes[0]->Grad1.size() != 0) {
    Grad1 += in_nodes[0]->Grad1;
    Grad2 += in_nodes[0]->Grad2;
  }
  if (in_nodes[1]->Grad1.size() != 0) {
    Grad1 += in_nodes[1]->Grad1;
    Grad2 += in_nodes[1]->Grad2;
  }
}

void Cholesky::compute_gradients() {
  // equation 19 and 20 of
  // Differentiation of the Cholesky decomposition by Iain Murray
  // https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
  assert(in_nodes.size() == 1);
  uint n = in_nodes[0]->value.type.rows;
  Eigen::MatrixXd L = value._matrix;
  Eigen::MatrixXd Sigma = in_nodes[0]->value._matrix;
  Grad1 = in_nodes[0]->Grad1;
  Grad2 = in_nodes[0]->Grad2;
  for (int i = 0; i < (int)n; i++) {
    // update first and second grad of L_d (index i, i)
    Eigen::VectorXd L_r = L(i, Eigen::seq(0, i - 1));
    Eigen::VectorXd dL_r = Grad1(i, Eigen::seq(0, i - 1));
    double dSigma_d = Grad1(i, i);
    double L_d = L(i, i);
    double dL_d = (dSigma_d / 2 - L_r.dot(dL_r)) / L_d;
    Grad1(i, i) = dL_d;
    // use product rule to get second derivative
    auto g = 1 / L_d;
    auto dg = -std::pow(L_d, -2) * dL_d;
    auto h = dSigma_d / 2 - dL_r.dot(L_r);
    auto dd_Sigma_d = Grad2(i, i);
    auto ddL_r = Grad2(i, Eigen::seq(0, i - 1));
    auto dh = dd_Sigma_d / 2 - ddL_r.dot(L_r) - dL_r.dot(dL_r);
    Grad2(i, i) = g * dh + dg * h;

    // update first and second grad of L_c (index i+1:N, i)
    Eigen::VectorXd dSigma_c = Grad1(Eigen::seq(i + 1, Eigen::last), i);
    Eigen::MatrixXd dL_B =
        Grad1(Eigen::seq(i + 1, Eigen::last), Eigen::seq(0, i - 1));
    Eigen::MatrixXd L_B =
        L(Eigen::seq(i + 1, Eigen::last), Eigen::seq(0, i - 1));
    Eigen::VectorXd L_c = L(Eigen::seq(i + 1, Eigen::last), i);
    Eigen::VectorXd dL_c =
        (dSigma_c - dL_B * L_r - L_B * dL_r - L_c * dL_d) / L_d;
    Grad1(Eigen::seq(i + 1, Eigen::last), i) = dL_c;
    // product rule for second derivative
    auto h_c = dSigma_c - dL_B * L_r - L_B * dL_r - L_c * dL_d;
    auto ddSigma_c = Grad2(Eigen::seq(i + 1, Eigen::last), i);
    auto ddL_B = Grad2(Eigen::seq(i + 1, Eigen::last), Eigen::seq(0, i - 1));
    auto ddL_d = Grad2(i, i);
    auto dh_c = ddSigma_c - (ddL_B * L_r + dL_B * dL_r) -
        (dL_B * dL_r + L_B * ddL_r.transpose()) - (dL_c * dL_d + L_c * ddL_d);
    Eigen::VectorXd ddL_c = g * dh_c + dg * h_c;
    Grad2(Eigen::seq(i + 1, Eigen::last), i) = ddL_c;
  }

  // zero out upper triangular
  for (uint i = 0; i < n; i++) {
    for (uint j = i + 1; j < n; j++) {
      Grad1(i, j) = 0.0;
      Grad2(i, j) = 0.0;
    }
  }
}

void MatrixExp::compute_gradients() {
  assert(in_nodes.size() == 1);
  // f(x) = e^g(x)
  // f'(x) = e^g(x) * g'(x)
  // f''(x) = e^g(x) * g'(x) * g'(x) + e^g(x) * g''(x)
  Grad1 = value._matrix.cwiseProduct(in_nodes[0]->Grad1);
  Grad2 = Grad1.cwiseProduct(in_nodes[0]->Grad1) +
      value._matrix.cwiseProduct(in_nodes[0]->Grad2);
}

void LogProb::compute_gradients() {
  auto dist = (Distribution*)in_nodes[0];
  auto value = in_nodes[1];

  // Compute the gradient with respect to the value.
  // Note that we assume the value itself is a function g(x) of some distant
  // variable x with respect to which we are computing the derivative.

  // First compute d/dg logprob(g) and d^2/dg^2 logprob(g),
  // which are the derivatives of the log probability of the distribution with
  // respect to the value parameter.
  double log_prob_value_grad1 = 0, log_prob_value_grad2 = 0;
  dist->gradient_log_prob_value(
      value->value, log_prob_value_grad1, log_prob_value_grad2);

  // Note: First order chain rule:
  // d/dx[f(g(x))]
  //     = f’(g(x)) g’(x)
  // Second order chain rule:
  // d^2/dx^2[f(g(x))]
  //     = d/dx[f’(g(x)) g’(x)] // first order chain rule
  //     = d/dx[f’(g(x))] g’(x) // product rule
  //       + d/dx[g’(x)] f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) // first order chain rule
  //       + g’’(x) f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) + g’’(x) f’(g(x))
  // Here f is log(PDF), g is the value of the parameter to the function
  // being differentiated, and g' and g'' are the incoming gradients of
  // the value.
  // We apply the first and second order chain rules to compute this node's
  // gradients (the component contributed by the value parameter).
  double result_grad1 = value->grad1 * log_prob_value_grad1;
  double result_grad2 = log_prob_value_grad2 * value->grad1 * value->grad1 +
      value->grad2 * log_prob_value_grad1;

  // Compute the gradient with respect to the parameters of the
  // distribution and add them to result_*. Note that the function
  // gradient_log_prob_param applies the chain rule so we do not have to do it.
  dist->gradient_log_prob_param(value->value, result_grad1, result_grad2);

  // Store the computed gradients.
  this->grad1 = result_grad1;
  this->grad2 = result_grad2;
}

void LogProb::backward() {
  auto dist = (Distribution*)in_nodes[0];
  auto value = in_nodes[1];
  auto adjunct = back_grad1;
  dist->backward_value(value->value, value->back_grad1, adjunct);
  dist->backward_param(value->value, adjunct);
}

} // namespace oper
} // namespace beanmachine
