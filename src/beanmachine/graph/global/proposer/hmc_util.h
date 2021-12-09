/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
  double update_step_size(double acceptance_prob);
  double finalize_step_size();

 private:
  double gamma;
  double t;
  double kappa;
  double optimal_acceptance_prob;
  double mu;
  double log_best_step_size;
  double closeness;
  int iteration;
};

} // namespace graph
} // namespace beanmachine
