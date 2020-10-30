// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <map>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

class Operator : public graph::Node {
 public:
  explicit Operator(graph::OperatorType op_type)
      : graph::Node(graph::NodeType::OPERATOR), op_type(op_type) {}
  ~Operator() override {}
  bool is_stochastic() const override {
    return false;
  }
  double log_prob() const override;
  void gradient_log_prob(double& grad1, double& grad2) const override;
  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override {}
  graph::OperatorType op_type;
};

class OperatorFactory {
 public:
  typedef std::unique_ptr<Operator> (*builder_type)(
      const std::vector<graph::Node*>&);

  OperatorFactory() = delete;
  ~OperatorFactory() {}

  static bool register_op(
      const graph::OperatorType op_type,
      builder_type op_builder);
  static std::unique_ptr<Operator> create_op(
      const graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes);

 private:
  static std::map<int, builder_type>& op_map();
};

} // namespace oper
} // namespace beanmachine
