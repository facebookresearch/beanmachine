// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <string>
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace factor {

class Factor : public graph::Node {
 public:
  static std::unique_ptr<Factor> new_factor(
      graph::FactorType fac_type,
      const std::vector<graph::Node*>& in_nodes);
  explicit Factor(graph::FactorType fac_type)
      : graph::Node(graph::NodeType::FACTOR), fac_type(fac_type) {}
  bool is_stochastic() const override {
    return true;
  }
  void eval(std::mt19937& /* gen */) override {}
  void compute_gradients(bool /* is_source_scalar */) override {}
  graph::FactorType fac_type;
};

} // namespace factor
} // namespace beanmachine
