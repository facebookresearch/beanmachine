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

/*
 * Transposes a non-scalar parent.
 */
class Transpose : public Operator {
 public:
  explicit Transpose(const std::vector<graph::Node*>& in_nodes);
  ~Transpose() override {}

  void eval(std::mt19937& gen) override;
  void backward() override {}
  void compute_gradients() override {}

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Transpose>(in_nodes);
  }
};

class MatrixMultiply : public Operator {
 public:
  explicit MatrixMultiply(const std::vector<graph::Node*>& in_nodes);
  ~MatrixMultiply() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixMultiply>(in_nodes);
  }
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
};

class ElementwiseMultiply : public Operator {
 public:
  explicit ElementwiseMultiply(const std::vector<graph::Node*>& in_nodes);
  ~ElementwiseMultiply() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<ElementwiseMultiply>(in_nodes);
  }
};

class MatrixAdd : public Operator {
 public:
  explicit MatrixAdd(const std::vector<graph::Node*>& in_nodes);
  ~MatrixAdd() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixAdd>(in_nodes);
  }
};

class MatrixNegate : public Operator {
 public:
  explicit MatrixNegate(const std::vector<graph::Node*>& in_nodes);
  ~MatrixNegate() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixNegate>(in_nodes);
  }
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
};

class Cholesky : public Operator {
 public:
  explicit Cholesky(const std::vector<graph::Node*>& in_nodes);
  ~Cholesky() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<Cholesky>(in_nodes);
  }
};

class MatrixExp : public Operator {
 public:
  explicit MatrixExp(const std::vector<graph::Node*>& in_nodes);
  ~MatrixExp() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixExp>(in_nodes);
  }
};

class MatrixSum : public Operator {
 public:
  explicit MatrixSum(const std::vector<graph::Node*>& in_nodes);
  ~MatrixSum() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixSum>(in_nodes);
  }

 private:
  static bool is_registered;
};

class MatrixLog : public Operator {
 public:
  explicit MatrixLog(const std::vector<graph::Node*>& in_nodes);
  ~MatrixLog() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixLog>(in_nodes);
  }
};

class MatrixLog1p : public Operator {
 public:
  explicit MatrixLog1p(const std::vector<graph::Node*>& in_nodes);
  ~MatrixLog1p() override {}

  void eval(std::mt19937& gen) override;
  void backward() override;
  void compute_gradients() override;

  static std::unique_ptr<Operator> new_op(
      const std::vector<graph::Node*>& in_nodes) {
    return std::make_unique<MatrixLog1p>(in_nodes);
  }
};

} // namespace oper
} // namespace beanmachine
