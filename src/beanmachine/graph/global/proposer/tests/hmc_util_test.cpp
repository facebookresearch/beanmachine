/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>

#include "beanmachine/graph/global/proposer/hmc_util.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, hmc_util_step_size) {
  StepSizeAdapter step_size_adapter = StepSizeAdapter(0.65);
  step_size_adapter.initialize(1.0);
  step_size_adapter.update_step_size(1, 0.5);
  EXPECT_NEAR(step_size_adapter.finalize_step_size(), 7.613, 1e-4);

  step_size_adapter.initialize(0.5);
  step_size_adapter.update_step_size(1, 0.0);
  step_size_adapter.update_step_size(2, 0.9);
  EXPECT_NEAR(step_size_adapter.finalize_step_size(), 1.7678, 1e-4);
}

TEST(testglobal, hmc_util_diag_cov) {
  /*
  Test diagonal covariance of
  [[0.0756, 0.5218],
  [0.2970, -0.4036],
  [-1.0120, -1.4934]]
  ->
  [[0.4909, 0.0],
  [0.0, 1.0175]]
  */
  DiagonalCovarianceComputer diag_cov = DiagonalCovarianceComputer();
  diag_cov.initialize(2);
  Eigen::VectorXd s1(2);
  s1 << 0.0756, 0.5218;
  diag_cov.update(s1);
  Eigen::VectorXd s2(2);
  s2 << 0.2970, -0.4036;
  diag_cov.update(s2);
  Eigen::VectorXd s3(2);
  s3 << -1.0120, -1.4934;
  diag_cov.update(s3);
  Eigen::MatrixXd covariance = diag_cov.finalize_updates();

  Eigen::MatrixXd expected_covariance(2, 2);
  expected_covariance << 0.4909, 0.0, 0.0, 1.0175;
  // Stan regularization
  double weight = 3.0 / (3.0 + 5.0);
  expected_covariance = weight * expected_covariance +
      1e-3 * (1 - weight) *
          Eigen::MatrixXd::Identity(
              expected_covariance.rows(), expected_covariance.cols());

  EXPECT_EQ(covariance.rows(), expected_covariance.rows());
  EXPECT_EQ(covariance.cols(), expected_covariance.cols());
  for (int i = 0; i < covariance.rows(); i++) {
    for (int j = 0; j < covariance.cols(); j++) {
      EXPECT_NEAR(covariance(i, j), expected_covariance(i, j), 1e-4);
    }
  }

  // test reset function
  diag_cov.reset();
  /*
  Test diagonal covariance of
  [[-0.2248,  0.2092],
  [-1.0136, -0.2332]]
  ->
  [[0.3111, 0.0],
  [0.0, 0.0979]]
  */
  s1 << -0.2248, 0.2092;
  diag_cov.update(s1);
  s2 << -1.0136, -0.2332;
  diag_cov.update(s2);
  covariance = diag_cov.finalize_updates();

  expected_covariance << 0.3111, 0.0, 0.0, 0.0979;
  // Stan regularization
  weight = 2.0 / (2.0 + 5.0);
  expected_covariance = weight * expected_covariance +
      1e-3 * (1 - weight) *
          Eigen::MatrixXd::Identity(
              expected_covariance.rows(), expected_covariance.cols());

  EXPECT_EQ(covariance.rows(), expected_covariance.rows());
  EXPECT_EQ(covariance.cols(), expected_covariance.cols());
  for (int i = 0; i < covariance.rows(); i++) {
    for (int j = 0; j < covariance.cols(); j++) {
      EXPECT_NEAR(covariance(i, j), expected_covariance(i, j), 1e-4);
    }
  }
}

void _expect_near_matrix(
    Eigen::MatrixXd& matrix1,
    Eigen::MatrixXd& matrix2,
    double delta) {
  EXPECT_EQ(matrix1.rows(), matrix2.rows());
  EXPECT_EQ(matrix1.cols(), matrix2.cols());
  for (int row = 0; row < matrix1.rows(); row++) {
    for (int col = 0; col < matrix1.cols(); col++) {
      EXPECT_NEAR(matrix1(row, col), matrix2(row, col), delta);
    }
  }
}

TEST(testglobal, hmc_util_windowed_mass_matrix) {
  // test that the identity mass matrix is returned until the 75th iteration
  WindowedMassMatrixAdapter mass_matrix_adapter = WindowedMassMatrixAdapter();
  mass_matrix_adapter.initialize(300, 3);
  Eigen::MatrixXd expected_mass_matrix;
  Eigen::VectorXd sample;
  Eigen::MatrixXd mass_matrix;
  bool window_end;

  for (int i = 1; i <= 75; i++) {
    sample = Eigen::VectorXd::Random(3);
    mass_matrix_adapter.update_mass_matrix(i, sample);
    window_end = mass_matrix_adapter.is_end_window(i);
    EXPECT_TRUE(mass_matrix.isIdentity());
    EXPECT_FALSE(window_end);
  }

  // adapt mass matrix for samples [76, 100)
  // mass matrix to use (returned by update_mass_matrix)
  // is still the identity matrix
  // use DiagonalCovarianceComputer to verify covariance
  DiagonalCovarianceComputer diag_cov = DiagonalCovarianceComputer();
  diag_cov.initialize(3);
  for (int i = 76; i <= 100; i++) {
    sample = Eigen::VectorXd::Random(3);
    mass_matrix_adapter.update_mass_matrix(i, sample);
    window_end = mass_matrix_adapter.is_end_window(i);
    if (window_end) {
      mass_matrix_adapter.get_mass_matrix_and_reset(i, mass_matrix);
    }
    // used for verification
    diag_cov.update(sample);
    if (i < 100) {
      EXPECT_TRUE(mass_matrix.isIdentity());
      EXPECT_FALSE(window_end);
    } else {
      // used for verification
      expected_mass_matrix = diag_cov.finalize_updates();
      _expect_near_matrix(expected_mass_matrix, mass_matrix, 1e-4);
      diag_cov.reset();
      EXPECT_TRUE(window_end);
    }
  }

  // adapt mass matrix for samples [101, 150]
  // mass matrix is the covariance from samples [76, 100],
  // which is verified using DiagonalCovarianceAlgorithm
  // until the last iteration, where it is updated
  for (int i = 101; i <= 150; i++) {
    sample = Eigen::VectorXd::Random(3);
    mass_matrix_adapter.update_mass_matrix(i, sample);
    window_end = mass_matrix_adapter.is_end_window(i);
    if (window_end) {
      mass_matrix_adapter.get_mass_matrix_and_reset(i, mass_matrix);
    }
    // used for verification
    diag_cov.update(sample);

    if (i < 150) {
      EXPECT_FALSE(window_end);
    } else {
      expected_mass_matrix = diag_cov.finalize_updates();
      diag_cov.reset();
      EXPECT_TRUE(window_end);
    }
    _expect_near_matrix(expected_mass_matrix, mass_matrix, 1e-4);
  }

  // adapt mass matrix for samples [150, 249]
  // mass matrix is the covariance from samples [101, 150]
  for (int i = 151; i <= 249; i++) {
    sample = Eigen::VectorXd::Random(3);
    mass_matrix_adapter.update_mass_matrix(i, sample);
    window_end = mass_matrix_adapter.is_end_window(i);
    if (window_end) {
      mass_matrix_adapter.get_mass_matrix_and_reset(i, mass_matrix);
    }
    _expect_near_matrix(expected_mass_matrix, mass_matrix, 1e-4);
  }
}
