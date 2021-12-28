/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <random>
#include <sstream>
#include <thread>
#include <variant>

#include "beanmachine/graph/double_matrix.h"

namespace beanmachine {
namespace graph {

// Conveniences for this compilation unit

using std::get;
using Matrix = DoubleMatrix::Matrix;

/// MatrixProperty methods

MatrixProperty::MatrixProperty(DoubleMatrix& owner) : owner(&owner) {}

inline Matrix& MatrixProperty::value() {
  return get<Matrix>(*owner);
}

inline const Matrix& MatrixProperty::value() const {
  return get<Matrix>(*owner);
}

Matrix& MatrixProperty::operator=(const Matrix& m) {
  owner->VariantBaseClass::operator=(m);
  return value();
}

MatrixProperty::operator const Matrix&() const {
  return value();
}

double MatrixProperty::coeff(Eigen::MatrixXd::Index i) const {
  return std::get<Eigen::MatrixXd>(*owner).coeff(i);
}

double& MatrixProperty::operator()(Eigen::MatrixXd::Index i) {
  return std::get<Eigen::MatrixXd>(*owner)(i);
}

double& MatrixProperty::operator()(
    Eigen::MatrixXd::Index row,
    Eigen::MatrixXd::Index col) {
  return std::get<Eigen::MatrixXd>(*owner)(row, col);
}

Eigen::MatrixXd::ColXpr MatrixProperty::col(Eigen::MatrixXd::Index i) {
  return std::get<Eigen::MatrixXd>(*owner).col(i);
}

double MatrixProperty::sum() {
  return value().sum();
}

Matrix& MatrixProperty::operator+=(const Matrix& increment) {
  return value() += increment;
}

Matrix& MatrixProperty::operator+=(const DoubleMatrix& increment) {
  return value() += get<Matrix>(increment);
}

Matrix& MatrixProperty::operator-=(const Matrix& increment) {
  return value() -= increment;
}

Matrix MatrixProperty::operator*(const Matrix& increment) {
  return value() * increment;
}

Matrix MatrixProperty::operator*(const DoubleMatrix& increment) {
  return value() * get<Matrix>(increment);
}

Matrix operator*(const Matrix& operand, const MatrixProperty& mp) {
  return operand * get<Matrix>(*mp.owner);
}

Matrix operator*(double operand, const MatrixProperty& mp) {
  return operand * get<Matrix>(*mp.owner);
}

Matrix::ColXpr operator+=(Matrix::ColXpr operand, const MatrixProperty& mp) {
  return operand += get<Matrix>(*mp.owner);
}

Eigen::MatrixXd& MatrixProperty::setZero(
    Eigen::MatrixXd::Index rows,
    Eigen::MatrixXd::Index cols) {
  if (not std::holds_alternative<Matrix>(*owner)) {
    *this = Eigen::MatrixXd();
  }
  return value().setZero(rows, cols);
}

Eigen::ArrayWrapper<Matrix> MatrixProperty::array() {
  return value().array();
}

Matrix::Scalar* MatrixProperty::data() {
  return value().data();
}

Matrix::Index MatrixProperty::size() {
  return value().size();
}

/// DoubleMatrix methods

#define TYPE(DM) ((DM).index())
#define DOUBLE 0
#define MATRIX 1

DoubleMatrixError error(const char* message) {
  return DoubleMatrixError(message);
}

DoubleMatrix::operator double() const {
  if (not std::holds_alternative<double>(*this)) {
    throw std::runtime_error(
        "operator double() on DoubleMatrix without double");
  }
  return std::get<double>(*this);
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

DoubleMatrix& DoubleMatrix::operator=(const DoubleMatrix& double_matrix) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      VariantBaseClass::operator=(get<double>(double_matrix));
      return *this;
    case MATRIX:
      VariantBaseClass::operator=(get<Matrix>(double_matrix));
      return *this;
    default:
      throw error("Assigning from DoubleMatrix without value");
  }
}

/// +=

DoubleMatrix& DoubleMatrix::operator+=(double d) {
  switch (TYPE(*this)) {
    case DOUBLE:
      get<double>(*this) += d;
      return *this;
    case MATRIX:
      throw error(
          "In-place addition of double to 'DoubleMatrix' containing matrix");
    default:
      throw error("In-place addition to empty DoubleMatrix");
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
      throw error("In-place addition to empty DoubleMatrix");
  }
}

/// *

DoubleMatrix operator*(const DoubleMatrix& double_matrix, double d) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return DoubleMatrix{get<double>(double_matrix) * d};
    case MATRIX:
      return DoubleMatrix{get<Matrix>(double_matrix) * d};
    default:
      throw error("Multiplying DoubleMatrix that does not hold a value.");
  }
}

DoubleMatrix operator*(double d, const DoubleMatrix& double_matrix) {
  return double_matrix * d;
}

Matrix operator*(const DoubleMatrix& double_matrix, const Matrix& arg) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return get<double>(double_matrix) * arg;
    case MATRIX:
      return get<Matrix>(double_matrix) * arg;
    default:
      throw error("Multiplying DoubleMatrix that does not hold a value.");
  }
}

Matrix operator*(const Matrix& arg, const DoubleMatrix& double_matrix) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return arg * get<double>(double_matrix);
    case MATRIX:
      return arg * get<Matrix>(double_matrix);
    default:
      throw error("Multiplying DoubleMatrix that does not hold a value.");
  }
}

DoubleMatrix operator*(const DoubleMatrix& dm1, const DoubleMatrix& dm2) {
  switch (TYPE(dm1)) {
    case DOUBLE:
      return get<double>(dm1) * dm2;
    case MATRIX:
      return DoubleMatrix{get<Matrix>(dm1) * dm2};
    default:
      throw error("Multiplying DoubleMatrix that does not hold a value.");
  }
}

/// +

// Note: Eigen does not support adding a matrix and a double, so
// here we can assume arguments will always contain information
// of the same type.

double operator+(const DoubleMatrix& double_matrix, double arg) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return get<double>(double_matrix) + arg;
    case MATRIX:
      throw error(
          "Adding DoubleMatrix with matrix to double is not supported by Eigen.");
    default:
      throw error("Adding DoubleMatrix that does not hold a value.");
  }
}

double operator+(double arg, const DoubleMatrix& double_matrix) {
  return double_matrix + arg;
}

DoubleMatrix operator+(
    const DoubleMatrix& double_matrix,
    const DoubleMatrix& arg) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      return DoubleMatrix{get<double>(double_matrix) + arg};
    case MATRIX:
      return DoubleMatrix{get<Matrix>(double_matrix) + get<Matrix>(arg)};
    default:
      throw error("Adding DoubleMatrix that does not hold a value.");
  }
}

Matrix operator+(const DoubleMatrix& double_matrix, const Matrix& arg) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      throw error(
          "Adding DoubleMatrix with double to a matrix is not supported by Eigen.");
    case MATRIX:
      return get<Matrix>(double_matrix) + arg;
    default:
      throw error("Adding DoubleMatrix that does not hold a value.");
  }
}

Matrix operator+(const Matrix& arg, const DoubleMatrix& double_matrix) {
  switch (TYPE(double_matrix)) {
    case DOUBLE:
      throw error(
          "Adding DoubleMatrix with double to a matrix is not supported by Eigen.");
    case MATRIX:
      return arg + get<Matrix>(double_matrix);
    default:
      throw error("Adding DoubleMatrix that does not hold a value.");
  }
}

#undef MATRIX
#undef DOUBLE
#undef TYPE

} // namespace graph
} // namespace beanmachine
