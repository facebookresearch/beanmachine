/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/double_matrix.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <thread>
#include <variant>

namespace beanmachine {
namespace graph {

// Conveniences for this compilation unit

using std::get;
using Matrix = DoubleMatrix::Matrix;

template <typename T>
bool has(const DoubleMatrix& double_matrix) {
  return std::holds_alternative<T>(double_matrix);
}

/// DoubleMatrix methods

#define TYPE(DM) ((DM).index())
#define DOUBLE 0
#define MATRIX 1

DoubleMatrixError double_matrix_error(const char* message) {
  return DoubleMatrixError(message);
}

/// Conversions

DoubleMatrix::operator double() const {
  if (not has<double>(*this)) {
    throw double_matrix_error(
        "operator double() on DoubleMatrix without double");
  }
  return get<double>(*this);
}

/// =

DoubleMatrix& DoubleMatrix::operator=(double d) {
  VariantBaseClass::operator=(d);
  return *this;
}

DoubleMatrix& DoubleMatrix::operator=(const Matrix& matrix) {
  VariantBaseClass::operator=(matrix);
  return *this;
}

DoubleMatrix& DoubleMatrix::operator=(const DoubleMatrix& another) {
  VariantBaseClass::operator=(static_cast<const VariantBaseClass&>(another));
  return *this;
}

DoubleMatrix& DoubleMatrix::operator=(DoubleMatrix&& another) {
  VariantBaseClass::operator=(static_cast<VariantBaseClass&&>(another));
  return *this;
}

/// array

DoubleMatrix::Array DoubleMatrix::array() {
  return get<Matrix>(*this).array();
}

const DoubleMatrix::ArrayOfConst DoubleMatrix::array() const {
  return get<Matrix>(*this).array();
}

/// +=

DoubleMatrix& DoubleMatrix::operator+=(double d) {
  switch (TYPE(*this)) {
    case DOUBLE:
      get<double>(*this) += d;
      return *this;
    case MATRIX:
      throw double_matrix_error(
          "In-place addition of double to 'DoubleMatrix' containing matrix");
    default:
      throw double_matrix_error("In-place addition to empty DoubleMatrix");
  }
}

DoubleMatrix& DoubleMatrix::operator+=(const Matrix& matrix) {
  switch (TYPE(*this)) {
    case DOUBLE:
      throw double_matrix_error(
          "In-place addition of matrix to 'DoubleMatrix' containing double");
    case MATRIX:
      get<Matrix>(*this) += matrix;
      return *this;
    default:
      throw double_matrix_error("In-place addition to empty DoubleMatrix");
  }
}

DoubleMatrix& DoubleMatrix::operator+=(const DoubleMatrix& another) {
  switch (TYPE(*this)) {
    case DOUBLE:
      get<double>(*this) += get<double>(another);
      return *this;
    case MATRIX:
      get<Matrix>(*this) += get<Matrix>(another);
      return *this;
    default:
      throw double_matrix_error("In-place addition to empty DoubleMatrix");
  }
}

/// -=

DoubleMatrix& DoubleMatrix::operator-=(double d) {
  switch (TYPE(*this)) {
    case DOUBLE:
      get<double>(*this) -= d;
      return *this;
    case MATRIX:
      throw double_matrix_error(
          "In-place subtraction of double to 'DoubleMatrix' containing matrix");
    default:
      throw double_matrix_error("In-place subtraction to empty DoubleMatrix");
  }
}

DoubleMatrix& DoubleMatrix::operator-=(const Matrix& matrix) {
  switch (TYPE(*this)) {
    case DOUBLE:
      throw double_matrix_error(
          "In-place subtraction of matrix to 'DoubleMatrix' containing double");
    case MATRIX:
      get<Matrix>(*this) -= matrix;
      return *this;
    default:
      throw double_matrix_error("In-place subtraction to empty DoubleMatrix");
  }
}

DoubleMatrix& DoubleMatrix::operator-=(const DoubleMatrix& another) {
  switch (TYPE(*this)) {
    case DOUBLE:
      get<double>(*this) -= get<double>(another);
      return *this;
    case MATRIX:
      get<Matrix>(*this) -= get<Matrix>(another);
      return *this;
    default:
      throw double_matrix_error("In-place subtraction to empty DoubleMatrix");
  }
}

/// operator() and coeff

Matrix::Scalar& DoubleMatrix::operator()(Index i) {
  return get<Matrix>(*this)(i);
}

Matrix::Scalar& DoubleMatrix::operator()(Index row, Index col) {
  return get<Matrix>(*this)(row, col);
}

Matrix::Scalar DoubleMatrix::operator()(Index i) const {
  return get<Matrix>(*this)(i);
}

Matrix::Scalar DoubleMatrix::operator()(Index row, Index col) const {
  return get<Matrix>(*this)(row, col);
}

const Matrix::Scalar& DoubleMatrix::coeff(
    Eigen::Index rolId,
    Eigen::Index colId) const {
  return get<Matrix>(*this).coeff(rolId, colId);
}

const Matrix::Scalar& DoubleMatrix::coeff(Eigen::Index index) const {
  return get<Matrix>(*this).coeff(index);
}

/// setZero

DoubleMatrix& DoubleMatrix::setZero(Index rows, Index cols) {
  if (not has<Matrix>(*this)) {
    *this = Matrix();
  }
  get<Matrix>(*this).setZero(rows, cols);
  return *this;
}

/// col

Matrix::ColXpr DoubleMatrix::col(Index i) {
  return get<Matrix>(*this).col(i);
}

Matrix::ColXpr operator+=(
    DoubleMatrix::Matrix::ColXpr col,
    const DoubleMatrix& double_matrix) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      throw double_matrix_error(
          "Adding column and double not supported by Eigen.");
    case MATRIX:
      return col += get<Matrix>(double_matrix);
    default:
      throw double_matrix_error(
          "Adding DoubleMatrix that does not hold a value to a column.");
  }
}

/// data

Matrix::Scalar* DoubleMatrix::data() {
  return get<Matrix>(*this).data();
}

/// size

Matrix::Index DoubleMatrix::size() {
  return get<Matrix>(*this).size();
}

/// sum

Matrix::Scalar DoubleMatrix::sum() {
  return get<Matrix>(*this).sum();
}

/// *

DoubleMatrix operator*(const DoubleMatrix& double_matrix, double d) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return DoubleMatrix{get<double>(double_matrix) * d};
    case MATRIX:
      return DoubleMatrix{get<Matrix>(double_matrix) * d};
    default:
      throw double_matrix_error(
          "Multiplying DoubleMatrix that does not hold a value.");
  }
}

DoubleMatrix operator*(double d, const DoubleMatrix& double_matrix) {
  return double_matrix * d;
}

DoubleMatrix operator*(const DoubleMatrix& dm1, const DoubleMatrix& dm2) {
  switch (TYPE(dm1)) {
    case DOUBLE:
      return get<double>(dm1) * dm2;
    case MATRIX:
      if (has<Matrix>(dm2)) {
        return DoubleMatrix{get<Matrix>(dm1) * get<Matrix>(dm2)};
      } else if (has<double>(dm2)) {
        return DoubleMatrix{get<Matrix>(dm1) * get<double>(dm2)};
      } else {
        throw double_matrix_error(
            "Multiplying DoubleMatrix that does not hold a value.");
      }
    default:
      throw double_matrix_error(
          "Multiplying DoubleMatrix that does not hold a value.");
  }
}

/// +

// Note: Eigen does not support adding a matrix and a double, so
// here we can assume arguments will always contain information
// of the same type.

double operator+(const DoubleMatrix& double_matrix, double d) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return get<double>(double_matrix) + d;
    case MATRIX:
      throw double_matrix_error(
          "Adding DoubleMatrix with matrix to double is not supported by Eigen.");
    default:
      throw double_matrix_error(
          "Adding DoubleMatrix that does not hold a value.");
  }
}

double operator+(double d, const DoubleMatrix& double_matrix) {
  return double_matrix + d;
}

Matrix operator+(const DoubleMatrix& double_matrix, const Matrix& matrix) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      throw double_matrix_error(
          "Adding DoubleMatrix with double to a matrix is not supported by Eigen.");
    case MATRIX:
      return get<Matrix>(double_matrix) + matrix;
    default:
      throw double_matrix_error(
          "Adding DoubleMatrix that does not hold a value.");
  }
}

Matrix operator+(const Matrix& matrix, const DoubleMatrix& double_matrix) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      throw double_matrix_error(
          "Adding DoubleMatrix with double to a matrix is not supported by Eigen.");
    case MATRIX:
      return matrix + get<Matrix>(double_matrix);
    default:
      throw double_matrix_error(
          "Adding DoubleMatrix that does not hold a value.");
  }
}

DoubleMatrix operator+(const DoubleMatrix& dm1, const DoubleMatrix& dm2) {
  switch (TYPE(dm1)) {
    case DOUBLE:
      return DoubleMatrix{get<double>(dm1) + dm2};
    case MATRIX:
      return DoubleMatrix{get<Matrix>(dm1) + get<Matrix>(dm2)};
    default:
      throw double_matrix_error(
          "Adding DoubleMatrix that does not hold a value.");
  }
}

#undef MATRIX
#undef DOUBLE
#undef TYPE

} // namespace graph
} // namespace beanmachine
