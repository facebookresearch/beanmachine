// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/operator/unaryop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/linalgop.h"

namespace beanmachine {
namespace oper {

// Note: that we use the following chain rule for the gradients of f(g(x))
// first: f'(g(x)) g'(x)
// second: f''(g(x)) g'(x)^2 + f'(g(x))g''(x)

void Complement::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  // for complement (f(y)=1-y) and negate(f(y)=-y): f'(y) = -1 and f''(y) = 0
  grad1 = -1 * in_nodes[0]->grad1;
  grad2 = -1 * in_nodes[0]->grad2;
}

void ToReal::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void ToPosReal::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  grad1 = in_nodes[0]->grad1;
  grad2 = in_nodes[0]->grad2;
}

void Negate::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  grad1 = -1 * in_nodes[0]->grad1;
  grad2 = -1 * in_nodes[0]->grad2;
}

void Exp::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  // for f(y) = exp(y) or f(y) = exp(y)-1 we have f'(y) = exp(y) and f''(y) =
  // exp(y)
  double exp_parent = std::exp(in_nodes[0]->value._double);
  grad1 = exp_parent * in_nodes[0]->grad1;
  grad2 = grad1 * in_nodes[0]->grad1 + exp_parent * in_nodes[0]->grad2;
}

void ExpM1::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  double exp_parent = std::exp(in_nodes[0]->value._double);
  grad1 = exp_parent * in_nodes[0]->grad1;
  grad2 = grad1 * in_nodes[0]->grad1 + exp_parent * in_nodes[0]->grad2;
}

void Log1pExp::compute_gradients(bool /* is_source_scalar */) {
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

void Log::compute_gradients(bool /* is_source_scalar */) {
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

void Phi::compute_gradients(bool /* is_source_scalar */) {
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

void Logistic::compute_gradients(bool /* is_source_scalar */) {
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

void NegativeLog::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 1);
  // f(x) = -log(x)
  // f'(x) = -1 / x
  // f''(x) = 1 / (x^2) = f'(x) * f'(x)
  double x = in_nodes[0]->value._double;
  double f_grad = -1.0 / x;
  double f_grad2 = f_grad * f_grad;
  grad1 = f_grad * in_nodes[0]->grad1;
  grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      f_grad * in_nodes[0]->grad2;
}

void Pow::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 2);
  // We wish to compute the first and second derivatives of x ** y.
  // Let g = y log x
  // Let f = exp g = x ** y
  // f'  = g' f
  // f'' = g'' f + g' f'
  // So we must compute g' and g''.
  // let m = y' log x
  // let n = x' y / x
  // g'  = m + n
  // g'' = m' + n'
  // m'  = y'' log x + x' y' / x
  // n'  = x'' y / x +
  //       x' y' / x -
  //       x' x' y / (x x)
  double f = value._double;
  double x = in_nodes[0]->value._double;
  double y = in_nodes[1]->value._double;
  double x1 = in_nodes[0]->grad1;
  double y1 = in_nodes[1]->grad1;
  double x2 = in_nodes[0]->grad2;
  double y2 = in_nodes[1]->grad2;
  double logx = std::log(x);
  double m = y1 * logx;
  double n = x1 * y / x;
  double g1 = m + n;
  double f1 = g1 * f;
  // m1 and n1 have a term in common; we can avoid computing it twice.
  double c = x1 * y1 / x;
  double m1 = y2 * logx + c;
  double n1 = x2 * y / x + c - x1 * x1 * y / (x * x);
  double g2 = m1 + n1;
  double f2 = g2 * f + g1 * f1;
  grad1 = f1;
  grad2 = f2;
}

void Add::compute_gradients(bool /* is_source_scalar */) {
  grad1 = grad2 = 0;
  for (const auto node : in_nodes) {
    grad1 += node->grad1;
    grad2 += node->grad2;
  }
}

void Multiply::compute_gradients(bool /* is_source_scalar */) {
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
  for (const auto node : in_nodes) {
    sum_product_one_grad2 *= node->value._double;
    sum_product_one_grad2 += product * node->grad2;
    sum_product_two_grad1 *= node->value._double;
    sum_product_two_grad1 += sum_product_one_grad1 * node->grad1;
    sum_product_one_grad1 *= node->value._double;
    sum_product_one_grad1 += product * node->grad1;
    product *= node->value._double;
  }
  grad1 = sum_product_one_grad1;
  grad2 = sum_product_two_grad1 * 2 + sum_product_one_grad2;
}

void LogSumExp::compute_gradients(bool /* is_source_scalar */) {
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
  const uint N = in_nodes.size();
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
    grad2 += f_grad[i] *
        (node_i->grad1 * (node_i->grad1 - grad1) + node_i->grad2);
  }
}

void IfThenElse::compute_gradients(bool /* is_source_scalar */) {
  assert(in_nodes.size() == 3);
  if (in_nodes[0]->value._bool) {
    grad1 = in_nodes[1]->grad1;
    grad2 = in_nodes[1]->grad2;
  } else {
    grad1 = in_nodes[2]->grad1;
    grad2 = in_nodes[2]->grad2;
  }
}

/*
For C = A @ B, where A is d1 by d2, B is d2 by d3, and the dimension of the
source node is d0. Consider the matrix multiplicatin as a function that
maps R^{d1*d2 + d2*d3} to R^{d1*d3}, where the input is constructed by
flattening A and B by column then concate them.
The first order gradient propagation is G_new = J @ G_parent, where:
J = (J_a, J_b) is d1*d3 by d1*d2 + d2*d3, and
G_parent = (G_a^T, G_b^T)^T is d1*d2 + d2*d3 by d0.
Thus G_new = J_a @ G_a + J_b @ G_b, where J_a and J_b are sparse:
J_a = B_{1,1} @ I, B_{2,1} @ I, ..., B_{d2,1} @ I
      B_{1,2} @ I, B_{2,2} @ I, ...
      ...                            B_{d2,d3} @ I,
where I is the d1 by d1 identity matrix,
J_b = A, 0, ..., 0
      0, A, 0, ...
      ...    A ...
      0, 0, ..., A
Thus the computation of G_new can be implemented by blocks.

Suppose the jth element of the flattened output C_j represents the matrix
element C(r,c), to propagate the 2nd order gradient of C_j w.r.t the source
node x at index i:
G2(j, i) = G_parent(,i)^T @ H(j) @ G_parent(,i) + J(j,) @ G2_parent(, i)
where H(j) is the Hessian matrix of C_j. Note that H(j) is very sparse,
with the only non-zero entry being d^2 C(r, c) / dA(r, d) dB(d, c) = 1 for d
in 1, 2, ..., d2. Thus we implement the calculation with nested loops on
the non-zero entries.
*/
void MatrixMultiply::compute_gradients(bool is_source_scalar) {
  assert(in_nodes.size() == 2);
  const Eigen::MatrixXd& Grad_a = in_nodes[0]->Grad1;
  const Eigen::MatrixXd& Grad_b = in_nodes[1]->Grad1;
  uint col_a = Grad_a.cols(), col_b = Grad_b.cols();
  if (col_a != col_b and col_a * col_b != 0) {
    throw std::runtime_error("source node dimension does not match");
  }
  uint d0 = std::max(col_a, col_b);
  if (d0 == 0) {
    // both input nodes are independent of source
    Grad1.setZero(0, 0);
    Grad2.setZero(0, 0);
    return;
  }
  if (is_source_scalar and d0 != 1) {
    throw std::runtime_error(
        "scalar source node dimension should be 1 but got " +
        std::to_string(d0));
  }
  uint d1 = in_nodes[0]->value.type.rows;
  uint d2 = in_nodes[0]->value.type.cols;
  uint d3 = in_nodes[1]->value.type.cols;
  Eigen::MatrixXd& A = in_nodes[0]->value._matrix;
  Eigen::MatrixXd& B = in_nodes[1]->value._matrix;

  Grad1 = Eigen::MatrixXd::Zero(d1 * d3, d0);
  double B_rc;
  if (col_a > 0) {
    for (uint c = 0; c < d3; c++) {
      for (uint r = 0; r < d2; r++) {
        // B_rc = (double)in_nodes[1]->value._matrix.coeff(r, c);
        B_rc = (double)(*(in_nodes[1]->value._matrix.data() + c * d2 + r));
        Grad1.block(c * d1, 0, d1, d0) += B_rc * Grad_a.block(r * d1, 0, d1, d0);
      }
    }
  }
  if (col_b > 0) {
    for (uint i = 0; i < d3; i++) {
      Grad1.block(i * d1, 0, d1, d0) += A * Grad_b.block(i * d2, 0, d2, d0);
    }
  }

  const Eigen::MatrixXd& Grad2_a = in_nodes[0]->Grad2;
  const Eigen::MatrixXd& Grad2_b = in_nodes[1]->Grad2;
  Grad2 = Eigen::MatrixXd::Zero(d1 * d3, d0);
  double* grad2_new;
  double grad_a, grad_b, b_dc, a_rd, grad2_a, grad2_b;
  for (uint i = 0; i < d0; i++) {
    for (uint c = 0; c < d3; c++) {
      for (uint r = 0; r < d1; r++) {
        // Grad2.coeff(c * d1 + r, i);
        grad2_new = Grad2.data() + i * d1 * d3 + c * d1 + r;
        for (uint d = 0; d < d2; d++) {
          if (col_a > 0) {
            // Grad_a.coeff(d * d1 + r, i);
            grad_a = *(Grad_a.data() + i * d1 * d2 + d * d1 + r);
            // Grad2_a.coeff(d * d1 + r, i);
            grad2_a = *(Grad2_a.data() + i * d1 * d2 + d * d1 + r);
          } else {
            grad_a = 0, grad2_a = 0;
          }
          if (col_b > 0) {
            // Grad_b.coeff(c * d2 + d, i);
            grad_b = *(Grad_b.data() + i * d2 * d3 + c * d2 + d);
            // Grad2_b.coeff(c * d2 + d, i);
            grad2_b = *(Grad2_b.data() + i * d2 * d3 + c * d2 + d);
          } else {
            grad_b = 0, grad2_b = 0;
          }
          // A.coeff(r, d);
          a_rd = *(A.data() + d * d1 + r);
          // B.coeff(d, c);
          b_dc = *(B.data() + c * d2 + d);

          *grad2_new += grad_a * grad_b * 2 + b_dc * grad2_a + a_rd * grad2_b;
        }
      }
    }
  }
}

} // namespace oper
} // namespace beanmachine
