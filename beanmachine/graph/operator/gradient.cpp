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
    // for f(y) = exp(y) or f(y) = exp(y)-1 we have f'(y) = exp(y) and f''(y) = exp(y)
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
      // the sum of all products with exactly one variable replaced with its second grad
      double sum_product_one_grad2 = 0.0;
      for (const auto node: in_nodes) {
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
      for (const auto node: in_nodes) {
        grad1 += node->grad1;
        grad2 += node->grad2;
      }
      break;
    }
    default: {
      throw std::runtime_error(
        "internal error: unexpected operator type "
        + std::to_string(static_cast<int>(op_type))
        + " at node_id " + std::to_string(index));
    }
  }
}

} // namespace oper
} // namespace beanmachine
