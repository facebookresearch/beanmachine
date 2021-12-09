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

class UnaryOperator : public Operator {
 public:
  UnaryOperator(
      graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes)
      : Operator(op_type) {
    if (in_nodes.size() != 1) {
      throw std::invalid_argument(
          "expecting exactly a single parent for unary operator " +
          std::to_string(static_cast<int>(op_type)));
    }
    // if the parent node's value type has not been initialized then we
    // can't define an operator here
    if (in_nodes[0]->value.type.atomic_type == graph::AtomicType::UNKNOWN or
        in_nodes[0]->value.type.variable_type == graph::VariableType::UNKNOWN) {
      throw std::invalid_argument(
          "unexpected parent node of type " +
          std::to_string(static_cast<int>(in_nodes[0]->node_type)) +
          " for operator type " + std::to_string(static_cast<int>(op_type)));
    }
  }
  ~UnaryOperator() override {}
  void eval(std::mt19937& /* gen */) override {}
  void compute_gradients() override {}
  void backward() override;
  virtual double jacobian() const {
    return 1.0;
  }
};

class Complement : public UnaryOperator {
 public:
  explicit Complement(const std::vector<graph::Node*>& in_nodes);
  ~Complement() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Complement>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToInt : public UnaryOperator {
 public:
  explicit ToInt(const std::vector<graph::Node*>& in_nodes);
  ~ToInt() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToInt>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToReal : public UnaryOperator {
 public:
  explicit ToReal(const std::vector<graph::Node*>& in_nodes);
  ~ToReal() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToReal>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToRealMatrix : public UnaryOperator {
 public:
  explicit ToRealMatrix(const std::vector<graph::Node*>& in_nodes);
  ~ToRealMatrix() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToRealMatrix>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToPosReal : public UnaryOperator {
 public:
  explicit ToPosReal(const std::vector<graph::Node*>& in_nodes);
  ~ToPosReal() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToPosReal>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToPosRealMatrix : public UnaryOperator {
 public:
  explicit ToPosRealMatrix(const std::vector<graph::Node*>& in_nodes);
  ~ToPosRealMatrix() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToPosRealMatrix>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToProbability : public UnaryOperator {
 public:
  explicit ToProbability(const std::vector<graph::Node*>& in_nodes);
  ~ToProbability() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToProbability>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ToNegReal : public UnaryOperator {
 public:
  explicit ToNegReal(const std::vector<graph::Node*>& in_nodes);
  ~ToNegReal() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ToNegReal>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Negate : public UnaryOperator {
 public:
  explicit Negate(const std::vector<graph::Node*>& in_nodes);
  ~Negate() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Negate>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Exp : public UnaryOperator {
 public:
  explicit Exp(const std::vector<graph::Node*>& in_nodes);
  ~Exp() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Exp>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ExpM1 : public UnaryOperator {
 public:
  explicit ExpM1(const std::vector<graph::Node*>& in_nodes);
  ~ExpM1() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ExpM1>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Phi : public UnaryOperator {
 public:
  explicit Phi(const std::vector<graph::Node*>& in_nodes);
  ~Phi() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Phi>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Logistic : public UnaryOperator {
 public:
  explicit Logistic(const std::vector<graph::Node*>& in_nodes);
  ~Logistic() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Logistic>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Log1pExp : public UnaryOperator {
 public:
  explicit Log1pExp(const std::vector<graph::Node*>& in_nodes);
  ~Log1pExp() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Log1pExp>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Log : public UnaryOperator {
 public:
  explicit Log(const std::vector<graph::Node*>& in_nodes);
  ~Log() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Log>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Log1mExp : public UnaryOperator {
 public:
  explicit Log1mExp(const std::vector<graph::Node*>& in_nodes);
  ~Log1mExp() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  double jacobian() const override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Log1mExp>(in_nodes);
  }

 private:
  static bool is_registered;
};

class LogSumExpVector : public UnaryOperator {
 public:
  explicit LogSumExpVector(const std::vector<graph::Node*>& in_nodes);
  ~LogSumExpVector() override {}

  void eval(std::mt19937& gen) override;
  void compute_gradients() override;
  void backward() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<LogSumExpVector>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
