/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

void test_conjugate_model_moments(
    GlobalMH& mh,
    std::vector<double> expected_moments,
    int num_samples = 5000,
    int num_warmup_samples = 2000,
    double delta = 0.02,
    int seed = 17);

std::vector<double> build_gamma_gamma_model(Graph& g);

std::vector<double> build_normal_normal_model(Graph& g);

std::vector<double> build_gamma_normal_model(Graph& g);

std::vector<double> build_beta_binomial_model(Graph& g);

std::vector<double> build_mixed_model(Graph& g);

void build_half_cauchy_model(Graph& g);
void test_half_cauchy_model(
    GlobalMH& mh,
    int num_samples = 5000,
    int num_warmup_samples = 2000,
    double delta = 0.02,
    int seed = 17);
} // namespace graph
} // namespace beanmachine
