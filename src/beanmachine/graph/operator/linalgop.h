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

class MatrixMultiply : public Operator {
 public:
  explicit MatrixMultiply(const std::vector<graph::Node*>& in_nodes);
  ~MatrixMultiply() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override {
    throw std::runtime_error(
        "MATRIX_MULTIPLY does not support forward gradient propagation.");
  }

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixMultiply>(in_nodes);
  }

 private:
  static bool is_registered;
};

class MatrixScale : public Operator {
 public:
  explicit MatrixScale(const std::vector<graph::Node*>& in_nodes);
  ~MatrixScale() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixScale>(in_nodes);
  }

 private:
  static bool is_registered;
};

class Index : public Operator {
 public:
  explicit Index(const std::vector<graph::Node*>& in_nodes);
  ~Index() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Index>(in_nodes);
  }

 private:
  static bool is_registered;
};

class ColumnIndex : public Operator {
 public:
  explicit ColumnIndex(const std::vector<graph::Node*>& in_nodes);
  ~ColumnIndex() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ColumnIndex>(in_nodes);
  }

 private:
  static bool is_registered;
};

class BroadcastAdd : public Operator {
 public:
  explicit BroadcastAdd(const std::vector<graph::Node*>& in_nodes);
  ~BroadcastAdd() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<BroadcastAdd>(in_nodes);
  }

 private:
  static bool is_registered;
};

} // namespace oper
} // namespace beanmachine
