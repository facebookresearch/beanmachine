// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/controlop.h"

namespace beanmachine {
namespace oper {

IfThenElse::IfThenElse(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::IF_THEN_ELSE) {
  if (in_nodes.size() != 3 or
      in_nodes[1]->value.type != in_nodes[2]->value.type) {
    throw std::invalid_argument(
        "operator IF_THEN_ELSE requires 3 args and arg2.type == arg3.type");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "operator IF_THEN_ELSE requires boolean first argument");
  }
  value = graph::AtomicValue(in_nodes[1]->value.type.atomic_type);
}

void IfThenElse::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 3);
  if (in_nodes[0]->value._bool) {
    value = in_nodes[1]->value;
  } else {
    value = in_nodes[2]->value;
  }
}

} // namespace oper
} // namespace beanmachine
