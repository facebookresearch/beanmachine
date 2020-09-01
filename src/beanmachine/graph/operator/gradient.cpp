// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace oper {

void Operator::compute_gradients() {
  switch (op_type) {
    case graph::OperatorType::SAMPLE: {
      // the gradient of a sample node means nothing
      break;
    }
    case graph::OperatorType::TO_REAL:
    case graph::OperatorType::TO_POS_REAL: {
      grad1 = in_nodes[0]->grad1;
      grad2 = in_nodes[0]->grad2;
      break;
    }
    // Note: that we use the following chain rule for the gradients of f(g(x))
    // first: f'(g(x)) g'(x)
    // second: f''(g(x)) g'(x)^2 + f'(g(x))g''(x)
    //
    // for complement (f(y)=1-y) and negate(f(y)=-y): f'(y) = -1 and f''(y) = 0
    case graph::OperatorType::COMPLEMENT:
    case graph::OperatorType::NEGATE: {
      grad1 = -1 * in_nodes[0]->grad1;
      grad2 = -1 * in_nodes[0]->grad2;
      break;
    }
    case graph::OperatorType::PHI: {
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
      break;
    }
    case graph::OperatorType::LOGISTIC: {
      // f(x) = 1 / (1 + exp(-x))
      // f'(x) = exp(-x) / (1 + exp(-x))^2 = f(x) * (1 - f(x))
      // f''(x) = f'(x) - 2 f(x) f'(x) = f'(x) (1 - 2 f(x))
      double f_x = value._double;
      double f_grad = f_x * (1 - f_x);
      double f_grad2 = f_grad * (1 - 2 * f_x);
      grad1 = f_grad * in_nodes[0]->grad1;
      grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
          f_grad * in_nodes[0]->grad2;
      break;
    }
    case graph::OperatorType::LOG: {
      // f(x) = log(x)
      // f'(x) = 1 / x
      // f''(x) = -1 / (x^2) = -f'(x) * f'(x)
      double x = in_nodes[0]->value._double;
      double f_grad = 1.0 / x;
      double f_grad2 = -f_grad * f_grad;
      grad1 = f_grad * in_nodes[0]->grad1;
      grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
          f_grad * in_nodes[0]->grad2;
      break;
    }
    case graph::OperatorType::NEGATIVE_LOG: {
      // f(x) = -log(x)
      // f'(x) = -1 / x
      // f''(x) = 1 / (x^2) = f'(x) * f'(x)
      double x = in_nodes[0]->value._double;
      double f_grad = -1.0 / x;
      double f_grad2 = f_grad * f_grad;
      grad1 = f_grad * in_nodes[0]->grad1;
      grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
          f_grad * in_nodes[0]->grad2;
      break;
    }
    case graph::OperatorType::LOG1PEXP: {
      // f(x) = log (1 + exp(x))
      // f'(x) = exp(x) / (1 + exp(x)) = 1 - exp(-f)
      // f''(x) = exp(x) / (1 + exp(x))^2 = f' * (1 - f')
      double f_x = value._double;
      double f_grad = 1.0 - std::exp(-f_x);
      double f_grad2 = f_grad * (1.0 - f_grad);
      grad1 = f_grad * in_nodes[0]->grad1;
      grad2 = f_grad2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
          f_grad * in_nodes[0]->grad2;
      break;
    }
    // for f(y) = exp(y) or f(y) = exp(y)-1 we have f'(y) = exp(y) and f''(y) =
    // exp(y)
    case graph::OperatorType::EXP:
    case graph::OperatorType::EXPM1: {
      double exp_parent = std::exp(in_nodes[0]->value._double);
      grad1 = exp_parent * in_nodes[0]->grad1;
      grad2 = grad1 * in_nodes[0]->grad1 + exp_parent * in_nodes[0]->grad2;
      break;
    }

    case graph::OperatorType::POW: {
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
      break;
    }

    case graph::OperatorType::MULTIPLY: {
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
      break;
    }
    case graph::OperatorType::ADD: {
      grad1 = grad2 = 0;
      for (const auto node : in_nodes) {
        grad1 += node->grad1;
        grad2 += node->grad2;
      }
      break;
    }
    case graph::OperatorType::LOGSUMEXP: {
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
      for (uint i = 0; i < N; i++) {
        const auto node_i = in_nodes[i];
        grad2 += f_grad[i] *
            (node_i->grad1 * (node_i->grad1 - grad1) + node_i->grad2);
      }
      break;
    }
    case graph::OperatorType::IF_THEN_ELSE: {
      if (in_nodes[0]->value._bool) {
        grad1 = in_nodes[1]->grad1;
        grad2 = in_nodes[1]->grad2;
      } else {
        grad1 = in_nodes[2]->grad1;
        grad2 = in_nodes[2]->grad2;
      }
      break;
    }
    default: {
      throw std::runtime_error(
          "internal error: unexpected operator type " +
          std::to_string(static_cast<int>(op_type)) + " at node_id " +
          std::to_string(index));
    }
  }
}

} // namespace oper
} // namespace beanmachine
