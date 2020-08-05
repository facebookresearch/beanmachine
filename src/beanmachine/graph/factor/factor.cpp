// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/factor/exp_product.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace factor {

std::unique_ptr<Factor> Factor::new_factor(
    graph::FactorType fac_type,
    const std::vector<graph::Node*>& in_nodes) {
  // check parent nodes are of the correct type
  for (graph::Node* parent : in_nodes) {
    if (parent->node_type != graph::NodeType::CONSTANT and
        parent->node_type != graph::NodeType::OPERATOR) {
      throw std::invalid_argument(
          "factor parents must be constant or operator");
    }
  }
  // now simply call the appropriate factor constructor
  if (fac_type == graph::FactorType::EXP_PRODUCT) {
    return std::make_unique<ExpProduct>(in_nodes);
  }
  throw std::invalid_argument(
      "Unknown factor " + std::to_string(static_cast<int>(fac_type)));
}

} // namespace factor
} // namespace beanmachine
