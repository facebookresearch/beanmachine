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

template <typename T>
inline bool has(const DoubleMatrix& dm) {
  return std::holds_alternative<T>(dm);
}

/// DoubleProperty methods

DoubleProperty::DoubleProperty(DoubleMatrix& owner) : owner(&owner) {}

inline double& DoubleProperty::value() {
  return get<double>(*owner);
}

inline const double& DoubleProperty::value() const {
  return get<double>(*owner);
}

double& DoubleProperty::operator=(const double& d) {
  owner->VariantBaseClass::operator=(d);
  return value();
}

DoubleProperty::operator double&() {
  return value();
}

DoubleProperty::operator const double&() const {
  return value();
}

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
  if (not has<Matrix>(*owner)) {
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

} // namespace graph
} // namespace beanmachine
