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

  double& value();

  const double& value() const;

  double& operator=(const double& d);

  operator double&();

  operator const double&() const;
};

class MatrixProperty {
 public:
  using Matrix = Eigen::MatrixXd;

  DoubleMatrix* owner;

  explicit MatrixProperty(DoubleMatrix& owner);

  Matrix& value();

  const Matrix& value() const;

  Matrix& operator=(const Matrix& m);

  operator const Matrix&() const;

  double coeff(Eigen::MatrixXd::Index i) const;

  double& operator()(Eigen::MatrixXd::Index i);

  double& operator()(Eigen::MatrixXd::Index row, Eigen::MatrixXd::Index col);

  Eigen::MatrixXd::ColXpr col(Eigen::MatrixXd::Index i);

  double sum();

  Matrix& operator+=(const Matrix& increment);

  Matrix& operator+=(const DoubleMatrix& increment);

  Matrix& operator-=(const Matrix& increment);

  Matrix operator*(const Matrix& operand);

  Matrix operator*(const DoubleMatrix& operand);

  Eigen::MatrixXd& setZero(
      Eigen::MatrixXd::Index rows,
      Eigen::MatrixXd::Index cols);

  Eigen::ArrayWrapper<Matrix> array();

  Matrix::Scalar* data();

  Matrix::Index size();
};

MatrixProperty::Matrix operator*(
    const MatrixProperty::Matrix& operand,
    const MatrixProperty& mp);

MatrixProperty::Matrix operator*(double operand, const MatrixProperty& mp);

MatrixProperty::Matrix::ColXpr operator+=(
    MatrixProperty::Matrix::ColXpr operand,
    const MatrixProperty& mp);

struct DoubleMatrix : public std::variant<double, MatrixProperty::Matrix> {
  using Matrix = MatrixProperty::Matrix;
  using VariantBaseClass = std::variant<double, Matrix>;

  DoubleProperty _double;
  MatrixProperty _matrix;

  DoubleMatrix() : _double(*this), _matrix(*this) {}
};

} // namespace graph
} // namespace beanmachine
