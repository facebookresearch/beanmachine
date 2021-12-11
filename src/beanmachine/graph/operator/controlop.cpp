/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/operator/controlop.h"

namespace beanmachine {
namespace oper {

IfThenElse::IfThenElse(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::IF_THEN_ELSE) {
  if (in_nodes.size() != 3) {
    throw std::invalid_argument(
        "operator IF_THEN_ELSE requires exactly three parents");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "operator IF_THEN_ELSE requires a boolean first parent");
  }
  if (in_nodes[1]->value.type != in_nodes[2]->value.type) {
    throw std::invalid_argument(
        "operator IF_THEN_ELSE requires that the second and third parents have the same type");
  }
  value = in_nodes[1]->value;
}

void IfThenElse::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 3);
  if (in_nodes[0]->value._bool) {
    value = in_nodes[1]->value;
  } else {
    value = in_nodes[2]->value;
  }
}

Choice::Choice(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::CHOICE) {
  if (in_nodes.size() < 2) {
    throw std::invalid_argument(
        "operator CHOICE requires at least two parents");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::NATURAL) {
    throw std::invalid_argument(
        "operator CHOICE requires a natural first parent");
  }
  graph::ValueType type1 = in_nodes[1]->value.type;
  for (uint i = 2; i < static_cast<uint>(in_nodes.size()); i += 1) {
    if (in_nodes[i]->value.type != type1) {
      throw std::invalid_argument(
          "operator CHOICE requires all parents except the first to have the same type");
    }
  }
  value = in_nodes[1]->value;
}

void Choice::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() >= 2);
  graph::natural_t choice = in_nodes[0]->value._natural + 1;
  if (choice > in_nodes.size()) {
    throw std::runtime_error(
        "invalid value for CHOICE operator at node_id " +
        std::to_string(index));
  }
  value = in_nodes[choice]->value;
}

} // namespace oper
} // namespace beanmachine
