/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace oper {

class MultiaryOperator : public Operator {
 public:
  MultiaryOperator(
      graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes)
      : Operator(op_type) {
    if (in_nodes.size() < 2) {
      throw std::invalid_argument(
          "expecting at least two parents for operator " +
          std::to_string(static_cast<int>(op_type)));
    }
    // all parent nodes should have the same atomic type
    graph::AtomicType type0 = in_nodes[0]->value.type.atomic_type;
    for (const graph::Node* node : in_nodes) {
      if (node->value.type.atomic_type != type0) {
        throw std::invalid_argument(
            "all parents of operator " +
            std::to_string(static_cast<int>(op_type)) +
            " should have the same atomic type");
      }
    }
  }
  ~MultiaryOperator() override {}
  void eval(std::mt19937& /* gen */) override {}
  void compute_gradients() override {}
  void backward() override {}
};

class Add : public MultiaryOperator {
 public:
  explicit Add(const std::vector<graph::Node*>& in_nodes);
  ~Add() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Add>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Multiply : public MultiaryOperator {
 public:
  explicit Multiply(const std::vector<graph::Node*>& in_nodes);
  ~Multiply() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Multiply>(in_nodes);
  }

 private:
  static bool is_registered;
};

class LogSumExp : public MultiaryOperator {
 public:
  explicit LogSumExp(const std::vector<graph::Node*>& in_nodes);
  ~LogSumExp() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<LogSumExp>(in_nodes);
  }

 private:
  static bool is_registered;
};

// Pow is a special multiary operator that does not require all parents
// to be of the same type
class Pow : public Operator {
 public:
  explicit Pow(const std::vector<graph::Node*>& in_nodes);
  ~Pow() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Pow>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToMatrix : public Operator {
 public:
  explicit ToMatrix(const std::vector<graph::Node*>& in_nodes);
  ~ToMatrix() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToMatrix>(in_nodes);
  }

 private:
  static bool is_registered;
};

class LogProb : public Operator {
 public:
  explicit LogProb(const std::vector<graph::Node*>& in_nodes);
  ~LogProb() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<LogProb>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
