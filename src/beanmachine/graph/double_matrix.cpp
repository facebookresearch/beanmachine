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

DoubleProperty::DoubleProperty(DoubleMatrix& owner) : owner(&owner) {}

double& DoubleProperty::operator=(const double& d) {
  owner->std::variant<double, Eigen::MatrixXd>::operator=(d);
  return std::get<double>(*owner);
}

DoubleProperty::operator double&() {
  return std::get<double>(*owner);
}

DoubleProperty::operator const double&() const {
  return std::get<double>(*owner);
}

MatrixProperty::MatrixProperty(DoubleMatrix& owner) : owner(&owner) {}

Eigen::MatrixXd& MatrixProperty::operator=(const Eigen::MatrixXd& m) {
  owner->std::variant<double, Eigen::MatrixXd>::operator=(m);
  return std::get<Eigen::MatrixXd>(*owner);
}

MatrixProperty::operator const Eigen::MatrixXd&() const {
  return std::get<Eigen::MatrixXd>(*owner);
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
  return std::get<Eigen::MatrixXd>(*owner).sum();
}

// template <typename Increment>
// Eigen::MatrixXd& MatrixProperty::operator+=(const Increment& increment) {
//   return std::get<Eigen::MatrixXd>(*owner) += increment;
// }

Eigen::MatrixXd& MatrixProperty::operator+=(const Eigen::MatrixXd& increment) {
  return std::get<Eigen::MatrixXd>(*owner) += increment;
}

Eigen::MatrixXd& MatrixProperty::operator+=(const DoubleMatrix& increment) {
  return std::get<Eigen::MatrixXd>(*owner) +=
      std::get<Eigen::MatrixXd>(increment);
}

Eigen::MatrixXd& MatrixProperty::operator-=(const Eigen::MatrixXd& increment) {
  return std::get<Eigen::MatrixXd>(*owner) -= increment;
}

Eigen::MatrixXd MatrixProperty::operator*(const Eigen::MatrixXd& increment) {
  return std::get<Eigen::MatrixXd>(*owner) * increment;
}

Eigen::MatrixXd MatrixProperty::operator*(const DoubleMatrix& increment) {
  return std::get<Eigen::MatrixXd>(*owner) *
      std::get<Eigen::MatrixXd>(increment);
}

Eigen::MatrixXd operator*(
    const Eigen::MatrixXd& operand,
    const MatrixProperty& mp) {
  return operand * std::get<Eigen::MatrixXd>(*mp.owner);
}

Eigen::MatrixXd operator*(double operand, const MatrixProperty& mp) {
  return operand * std::get<Eigen::MatrixXd>(*mp.owner);
}

Eigen::MatrixXd::ColXpr operator+=(
    Eigen::MatrixXd::ColXpr operand,
    const MatrixProperty& mp) {
  return operand += std::get<Eigen::MatrixXd>(*mp.owner);
}

Eigen::MatrixXd& MatrixProperty::setZero(
    Eigen::MatrixXd::Index rows,
    Eigen::MatrixXd::Index cols) {
  if (not std::holds_alternative<Eigen::MatrixXd>(*owner)) {
    *this = Eigen::MatrixXd();
  }
  return std::get<Eigen::MatrixXd>(*owner).setZero(rows, cols);
}

Eigen::ArrayWrapper<Eigen::MatrixXd> MatrixProperty::array() {
  return std::get<Eigen::MatrixXd>(*owner).array();
}

Eigen::MatrixXd::Scalar* MatrixProperty::data() {
  return std::get<Eigen::MatrixXd>(*owner).data();
}

Eigen::MatrixXd::Index MatrixProperty::size() {
  return std::get<Eigen::MatrixXd>(*owner).size();
}

} // namespace graph
} // namespace beanmachine
