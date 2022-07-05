/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class StepSizeAdapter {
 public:
  explicit StepSizeAdapter(double optimal_acceptance_prob);
  void initialize(double step_size);
  double update_step_size(int iteration, double acceptance_prob);
  double finalize_step_size();

 private:
  double gamma;
  double t;
  double kappa;
  double optimal_acceptance_prob;
  double mu;
  double log_best_step_size;
  double closeness;
};

class DiagonalCovarianceComputer {
 public:
  DiagonalCovarianceComputer() {}
  void initialize(int size);
  void reset();
  void update(Eigen::VectorXd sample);
  Eigen::MatrixXd finalize_updates();

 private:
  int iteration;
  Eigen::MatrixXd sample_mean;
  Eigen::MatrixXd M2;
  Eigen::MatrixXd covariance;
};

class WindowedMassMatrixAdapter {
  /*
  Adaptation of Automatic Parameter Tuning from Stan
  https://mc-stan.org/docs/2_27/reference-manual/hmc-algorithm-parameters.html
  */
 public:
  WindowedMassMatrixAdapter() {}
  void initialize(int num_warmup_samples, int size);
  bool is_end_window(int iteration);
  void update_mass_matrix(int iteration, Eigen::VectorXd sample);
  void get_mass_matrix_and_reset(int iteration, Eigen::MatrixXd& mass_inv);

 private:
  DiagonalCovarianceComputer cov_alg;
  int start_window_iter;
  int end_adaptation_iter;

  int window_size;
  Eigen::MatrixXd mass_inv;
};

} // namespace graph
} // namespace beanmachine
