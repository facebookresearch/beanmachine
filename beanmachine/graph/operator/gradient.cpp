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
    case graph::OperatorType::TO_POS_REAL:
    case graph::OperatorType::TO_TENSOR: {
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
    // for f(y) = exp(y) or f(y) = exp(y)-1 we have f'(y) = exp(y) and f''(y) =
    // exp(y)
    case graph::OperatorType::EXP:
    case graph::OperatorType::EXPM1: {
      double exp_parent = std::exp(in_nodes[0]->value._double);
      grad1 = exp_parent * in_nodes[0]->grad1;
      grad2 = grad1 * in_nodes[0]->grad1 + exp_parent * in_nodes[0]->grad2;
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
