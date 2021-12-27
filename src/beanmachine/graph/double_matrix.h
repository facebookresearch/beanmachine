/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include <variant>

namespace beanmachine {
namespace graph {

struct DoubleMatrix;

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

  MatrixProperty _matrix;

  DoubleMatrix() : _matrix(*this) {}

  explicit DoubleMatrix(double d) : VariantBaseClass(d), _matrix(*this) {}

  explicit DoubleMatrix(Eigen::MatrixXd matrix)
      : VariantBaseClass(matrix), _matrix(*this) {}

  operator double() const;

  DoubleMatrix& operator=(double d);
  DoubleMatrix& operator=(const Matrix& d);
  DoubleMatrix& operator=(const DoubleMatrix& d);

  DoubleMatrix& operator+=(double d);
  DoubleMatrix& operator+=(const DoubleMatrix& another);
};

/// *

DoubleMatrix operator*(const DoubleMatrix& double_matrix, double arg);

DoubleMatrix operator*(double arg, const DoubleMatrix& double_matrix);

DoubleMatrix::Matrix operator*(
    const DoubleMatrix& double_matrix,
    const DoubleMatrix::Matrix& arg);

DoubleMatrix::Matrix operator*(
    const DoubleMatrix::Matrix& arg,
    const DoubleMatrix& double_matrix);

DoubleMatrix operator*(
    const DoubleMatrix& double_matrix,
    const DoubleMatrix& arg);

/// +

double operator+(const DoubleMatrix& double_matrix, double arg);

double operator+(double arg, const DoubleMatrix& double_matrix);

DoubleMatrix::Matrix operator+(
    const DoubleMatrix::Matrix& arg,
    const DoubleMatrix& double_matrix);

DoubleMatrix::Matrix operator+(
    const DoubleMatrix& double_matrix,
    const DoubleMatrix::Matrix& arg);

DoubleMatrix::Matrix operator+(
    const DoubleMatrix::Matrix& arg,
    const DoubleMatrix& double_matrix);

DoubleMatrix operator+(
    const DoubleMatrix& double_matrix,
    const DoubleMatrix& arg);

struct DoubleMatrixError : public std::runtime_error {
  explicit DoubleMatrixError(const char* message)
      : std::runtime_error(message) {}
};

} // namespace graph
} // namespace beanmachine
