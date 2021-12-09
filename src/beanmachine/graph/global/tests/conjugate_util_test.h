/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

void build_gamma_gamma_model(Graph& g);
void test_gamma_gamma_model(
    GlobalMH& mh,
    int num_samples = 5000,
    int seed = 17,
    int num_warmup_samples = 0,
    double delta = 0.01);

void build_normal_normal_model(Graph& g);
void test_normal_normal_model(
    GlobalMH& mh,
    int num_samples = 5000,
    int seed = 17,
    int num_warmup_samples = 0,
    double delta = 0.01);

} // namespace graph
} // namespace beanmachine
