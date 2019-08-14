// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <string>
#include <beanmachine/graph/graph.h>

namespace beanmachine {
namespace oper {

class Operator : public graph::Node {
 public:
  Operator(
      graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Operator() override {}
  void eval(std::mt19937& gen) override;
  graph::OperatorType op_type;
};

} // namespace oper
} // namespace beanmachine
