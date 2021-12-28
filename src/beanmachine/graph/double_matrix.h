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

struct DoubleMatrix : public std::variant<double, Eigen::MatrixXd> {
  using Matrix = Eigen::MatrixXd;
  using Index = Matrix::Index;
  using VariantBaseClass = std::variant<double, Matrix>;
  using Array = Eigen::ArrayWrapper<Matrix>;
  using ArrayOfConst = Eigen::ArrayWrapper<const Matrix>;

  DoubleMatrix() {}

  explicit DoubleMatrix(double d) : VariantBaseClass(d) {}

  explicit DoubleMatrix(Eigen::MatrixXd matrix) : VariantBaseClass(matrix) {}

  DoubleMatrix(const DoubleMatrix& another)
      : VariantBaseClass(static_cast<const VariantBaseClass&>(another)) {}

  DoubleMatrix(DoubleMatrix&& another)
      : VariantBaseClass(static_cast<VariantBaseClass&&>(another)) {}

  /* implicit */ operator double() const;

  double& as_double() {
    return std::get<double>(*this);
  }

  double as_double() const {
    return std::get<double>(*this);
  }

  Matrix& as_matrix() {
    return std::get<Matrix>(*this);
  }

  const Matrix& as_matrix() const {
    return std::get<Matrix>(*this);
  }

  DoubleMatrix& operator=(double d);
  DoubleMatrix& operator=(const Matrix& d);
  DoubleMatrix& operator=(const DoubleMatrix& d);
  DoubleMatrix& operator=(DoubleMatrix&& another);

  DoubleMatrix& operator+=(double d);
  DoubleMatrix& operator+=(const Matrix& matrix);
  DoubleMatrix& operator+=(const DoubleMatrix& another);

  DoubleMatrix& operator-=(double d);
  DoubleMatrix& operator-=(const Matrix& matrix);
  DoubleMatrix& operator-=(const DoubleMatrix& another);

  // A substitute for operator*(DoubleMatrix, Matrix).
  //
  // One might ask why we need these method instead of operator*(DoubleMatrix,
  // Matrix).
  // The short story: This is needed because we provide implicit
  // DoubleMatrix -> double conversion, which makes operator*(DoubleMatrix,
  // Matrix) ambiguous with operator*(double, Matrix). While this could be
  // avoided by disabling implicit double conversion, that would make it not
  // possible to write things like:
  // double x = double_matrix;
  // equals(double_matrix, 0.9, 0.0001);
  // which are much more common that multiplication by a matrix.
  //
  // The longer story:
  // In fact, if we define operator*(DoubleMatrix, Matrix) directly,
  // it does take precedence over implicit double conversion and there is no
  // problem. However that hurts performance because it forces premature
  // evaluation of matrix expression objects in Eigen to Matrix,
  // as explained in
  // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html. To
  // avoid this performance hit we need to define template <typename Derived>
  // operator*(DoubleMatrix, Eigen::MatrixBase<Derived>) but that does *not*
  // take precedence over operator*(double, Matrix).
  //
  // Another possibility is to use a static_cast before multiplying,
  // but this would prevent generic code that works for both doubles and
  // matrices.
  template <typename Derived>
  static DoubleMatrix times(
      const DoubleMatrix& dm,
      const Eigen::MatrixBase<Derived>& matrix) {
    switch (dm.index()) {
      case 0:
        return DoubleMatrix{std::get<double>(dm) * matrix};
      case 1:
        return DoubleMatrix{std::get<Matrix>(dm) * matrix};
      default:
        throw std::runtime_error(
            "Multiplying DoubleMatrix that does not hold a value.");
    }
  }

  // A substitute for operator*(Matrix, DoubleMatrix).
  template <typename Derived>
  static DoubleMatrix times(
      const Eigen::MatrixBase<Derived>& matrix,
      const DoubleMatrix& dm) {
    switch (dm.index()) {
      case 0:
        return DoubleMatrix{matrix * std::get<double>(dm)};
      case 1:
        return DoubleMatrix{matrix * std::get<Matrix>(dm)};
      default:
        throw std::runtime_error(
            "Multiplying DoubleMatrix that does not hold a value.");
    }
  }

  Array array();
  const ArrayOfConst array() const;

  double& operator()(Index i);
  double& operator()(Index row, Index col);
  double operator()(Index i) const;
  double operator()(Index row, Index col) const;

  const Matrix::Scalar& coeff(Index rolId, Index colId) const;
  const Matrix::Scalar& coeff(Index index) const;

  /* Resizes to the given size, and sets all coefficients in this expression to
   * zero. */
  DoubleMatrix& setZero(Index rows, Index cols);

  Matrix::ColXpr col(Index i);

  Matrix::Scalar* data();

  Matrix::Index size();

  Matrix::Scalar sum();
};

DoubleMatrix::Matrix::ColXpr operator+=(
    DoubleMatrix::Matrix::ColXpr col,
    const DoubleMatrix& double_matrix);

/// *

DoubleMatrix operator*(const DoubleMatrix& double_matrix, double d);

DoubleMatrix operator*(double d, const DoubleMatrix& double_matrix);

template <typename Derived>
DoubleMatrix operator*(
    const DoubleMatrix& double_matrix,
    const Eigen::MatrixBase<Derived>& matrix) {
  switch (double_matrix.index()) {
    case 0:
      return DoubleMatrix{std::get<double>(double_matrix) * matrix};
    case 1:
      return DoubleMatrix{
          std::get<DoubleMatrix::Matrix>(double_matrix) * matrix};
    default:
      throw std::runtime_error(
          "Multiplying DoubleMatrix that does not hold a value.");
  }
}

template <typename Derived>
DoubleMatrix operator*(
    const Eigen::MatrixBase<Derived>& matrix,
    const DoubleMatrix& double_matrix) {
  switch (double_matrix.index()) {
    case 0:
      return DoubleMatrix{matrix * std::get<double>(double_matrix)};
    case 1:
      return DoubleMatrix{
          matrix * std::get<DoubleMatrix::Matrix>(double_matrix)};
    default:
      throw std::runtime_error(
          "Multiplying DoubleMatrix that does not hold a value.");
  }
}

DoubleMatrix operator*(const DoubleMatrix& dm1, const DoubleMatrix& dm2);

/// +

double operator+(const DoubleMatrix& double_matrix, double d);

double operator+(double d, const DoubleMatrix& double_matrix);

DoubleMatrix::Matrix operator+(
    const DoubleMatrix::Matrix& matrix,
    const DoubleMatrix& double_matrix);

DoubleMatrix::Matrix operator+(
    const DoubleMatrix& double_matrix,
    const DoubleMatrix::Matrix& matrix);

DoubleMatrix::Matrix operator+(
    const DoubleMatrix::Matrix& matrix,
    const DoubleMatrix& double_matrix);

DoubleMatrix operator+(const DoubleMatrix& dm1, const DoubleMatrix& dm2);

struct DoubleMatrixError : public std::runtime_error {
  explicit DoubleMatrixError(const char* message)
      : std::runtime_error(message) {}
};

} // namespace graph
} // namespace beanmachine
