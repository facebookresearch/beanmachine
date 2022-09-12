/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES

#include "beanmachine/graph/distribution/product.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;
using namespace util;
using namespace std;

invalid_argument different_sample_types(
    const ValueType& type1,
    const ValueType& type2) {
  return invalid_argument(
      string(
          "Product distribution received distributions with different sample types: ") +
      type1.to_string() + " and " + type2.to_string());
}

invalid_argument at_least_one_parent() {
  return invalid_argument("Product distribution must have at least one parent");
}

invalid_argument
sample_type_from_parents_does_not_agree_with_required_sample_type(
    const AtomicType& sample_type_from_parents,
    const ValueType& required_sample_type) {
  return invalid_argument(
      string("Sample type from parents ") +
      string(NAMEOF_ENUM(sample_type_from_parents)) +
      " does not agree with required sample type " +
      required_sample_type.to_string());
}

ValueType get_unique_sample_type(const vector<Node*>& in_nodes) {
  // This function causes a false positive Uninitilized Value error by Infer.
  // It says field variable_type of ValueType may be uninitialized,
  // but all constructors of that type initialize that field.
  return get_unique_element_if_any_or_throw_exceptions(
      sample_type_iterator(in_nodes.begin()),
      sample_type_iterator(in_nodes.end()),
      at_least_one_parent,
      different_sample_types);
}

//////////////// Product methods

Product::Product(const vector<Node*>& in_nodes)
    : Distribution(DistributionType::PRODUCT, get_unique_sample_type(in_nodes)),
      in_distributions(vector_dynamic_cast<Distribution>(in_nodes)) {}

Product::Product(AtomicType sample_type, const vector<Node*>& in_nodes)
    : Distribution(DistributionType::PRODUCT, get_unique_sample_type(in_nodes)),
      in_distributions(vector_dynamic_cast<Distribution>(in_nodes)) {
  check_required_sample_type_against_sample_type_from_parents(sample_type);
}

void Product::check_required_sample_type_against_sample_type_from_parents(
    AtomicType sample_type) {
  if (!atomic_type_unknown_or_equal_to(sample_type, this->sample_type)) {
    throw sample_type_from_parents_does_not_agree_with_required_sample_type(
        sample_type, this->sample_type);
  }
}

double Product::log_prob(const NodeValue& value) const {
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    auto log_probs = map(in_distributions, log_prob_getter(value));
    return sum(log_probs);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    return log_prob_iid(value).sum();
  } else {
    throw std::runtime_error(
        "Product::log_prob applied to invalid variable type");
  }
}

void Product::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  in_distributions[0]->log_prob_iid(value, log_probs);
  for (size_t i = 1; i != in_distributions.size(); i++) {
    log_probs.array() += in_distributions[i]->log_prob_iid(value).array();
  }
}

void Product::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  for (auto in_distribution : in_distributions) {
    in_distribution->gradient_log_prob_value(value, grad1, grad2);
  }
}

void Product::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  for (auto in_distribution : in_distributions) {
    in_distribution->gradient_log_prob_param(value, grad1, grad2);
  }
}

void Product::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (adjunct == 0.0) {
    return;
  }
  for (auto in_distribution : in_distributions) {
    in_distribution->backward_value(value, back_grad, adjunct);
  }
}

void Product::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  for (auto in_distribution : in_distributions) {
    in_distribution->backward_value_iid(value, back_grad);
  }
}

void Product::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  for (auto in_distribution : in_distributions) {
    in_distribution->backward_value_iid(value, back_grad, adjunct);
  }
}

void Product::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (adjunct == 0.0) {
    return;
  }
  for (auto in_distribution : in_distributions) {
    in_distribution->backward_param(value, adjunct);
  }
}

void Product::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  for (auto in_distribution : in_distributions) {
    in_distribution->backward_param_iid(value);
  }
}

void Product::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  for (auto in_distribution : in_distributions) {
    in_distribution->backward_param_iid(value, adjunct);
  }
}

} // namespace distribution
} // namespace beanmachine
