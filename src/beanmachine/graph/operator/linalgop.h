// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace oper {

class MatrixMultiply : public Operator {
 public:
  explicit MatrixMultiply(const std::vector<graph::Node*>& in_nodes);
  ~MatrixMultiply() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients(bool is_source_scalar) override;

  static std::unique_ptr<Operator> new_op(const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixMultiply>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
