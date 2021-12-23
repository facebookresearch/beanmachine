/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <Eigen/Dense>
#include <string>
#include <variant>

namespace beanmachine {
namespace graph {

struct DoubleMatrix;

class DoubleProperty {
 public:
  DoubleMatrix* owner;

  explicit DoubleProperty(DoubleMatrix& owner);

  double& operator=(const double& d);

  operator double&();

  operator const double&() const;
};

class MatrixProperty {
 public:
  DoubleMatrix* owner;

  explicit MatrixProperty(DoubleMatrix& owner);

  Eigen::MatrixXd& operator=(const Eigen::MatrixXd& m);

  operator const Eigen::MatrixXd&() const;

  double coeff(Eigen::MatrixXd::Index i) const;

  double& operator()(Eigen::MatrixXd::Index i);

  double& operator()(Eigen::MatrixXd::Index row, Eigen::MatrixXd::Index col);

  Eigen::MatrixXd::ColXpr col(Eigen::MatrixXd::Index i);

  double sum();

  // template <typename Increment>
  // Eigen::MatrixXd& operator+=(const Increment& increment);

  Eigen::MatrixXd& operator+=(const Eigen::MatrixXd& increment);

  Eigen::MatrixXd& operator+=(const DoubleMatrix& increment);

  Eigen::MatrixXd& operator-=(const Eigen::MatrixXd& increment);

  Eigen::MatrixXd operator*(const Eigen::MatrixXd& operand);

  Eigen::MatrixXd operator*(const DoubleMatrix& operand);

  Eigen::MatrixXd& setZero(
      Eigen::MatrixXd::Index rows,
      Eigen::MatrixXd::Index cols);

  Eigen::ArrayWrapper<Eigen::MatrixXd> array();

  Eigen::MatrixXd::Scalar* data();

  Eigen::MatrixXd::Index size();
};

Eigen::MatrixXd operator*(
    const Eigen::MatrixXd& operand,
    const MatrixProperty& mp);

Eigen::MatrixXd operator*(double operand, const MatrixProperty& mp);

Eigen::MatrixXd::ColXpr operator+=(
    Eigen::MatrixXd::ColXpr operand,
    const MatrixProperty& mp);

struct DoubleMatrix : public std::variant<double, Eigen::MatrixXd> {
  DoubleProperty _double;
  MatrixProperty _matrix;

  DoubleMatrix() : _double(*this), _matrix(*this) {}
};

} // namespace graph
} // namespace beanmachine
