/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <typeinfo>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/testing_util.h"

namespace beanmachine::util {

using namespace std;
using namespace graph;

void test_nmc_against_nuts(
    Graph& graph,
    int num_rounds,
    int num_samples,
    int warmup_samples,
    function<unsigned()> seed_getter,
    function<void(double, double)> tester) {
  if (graph.queries.empty()) {
    throw invalid_argument(
        "test_nmc_against_nuts requires at least one query in graph.");
  }
  auto measured_max_abs_mean_diff = 0.0;
  for (int i = 0; i != num_rounds; i++) {
    auto seed = seed_getter();

    auto means_nmc = graph.infer_mean(num_samples, InferenceType::NMC, seed);

    NUTS nuts = NUTS(graph);
    auto samples = nuts.infer(num_samples, seed, warmup_samples);
    auto means_nuts = compute_means(samples);

    assert(!means_nmc.empty());
    assert(!means_nuts.empty());

    tester(means_nmc[0], means_nuts[0]);

    auto abs_diff = std::abs(means_nmc[0] - means_nuts[0]);
    if (abs_diff > measured_max_abs_mean_diff) {
      measured_max_abs_mean_diff = abs_diff;
    }

    cout << "NMC  result: " << means_nmc[0] << endl;
    cout << "NUTS result: " << means_nuts[0] << endl;
  }
  cout << "Measured max absolute difference: " << measured_max_abs_mean_diff
       << endl;
}

double compute_mean_at_index(vector<vector<NodeValue>> samples, size_t index) {
  double mean = 0;
  for (size_t i = 0; i < samples.size(); i++) {
    assert(samples[i].size() > index);
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    mean += samples[i][index]._double;
  }
  mean /= samples.size();
  return mean;
}

vector<double> compute_means(vector<vector<NodeValue>> samples) {
  if (samples.empty()) {
    return vector<double>();
  }
  auto num_dims = samples[0].size();
  auto means = vector<double>(num_dims);
  for (size_t i = 0; i != num_dims; i++) {
    means[i] = compute_mean_at_index(samples, i);
  }
  return means;
}

} // namespace beanmachine::util
