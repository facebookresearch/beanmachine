/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/nmc.h"
#include "beanmachine/graph/nmc_stepper.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/nmc_dirichlet_beta_single_site_stepping_method.h"
#include "beanmachine/graph/stepper/single_site/nmc_dirichlet_gamma_single_site_stepping_method.h"
#include "beanmachine/graph/stepper/single_site/nmc_scalar_single_site_stepping_method.h"
#include "beanmachine/graph/stepper/single_site/sequential_single_site_stepper.h"
#include "beanmachine/graph/stepper/single_site/single_site_stepping_method.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

NMC::NMC(Graph* graph, uint seed) : MH(graph, seed, new NMCStepper(this)) {}
// Ok to allocate and not delete NMCStepper because MH takes ownership
// of its stepper.

NMC::~NMC() {}

std::string NMC::is_not_supported(Node* node) {
  if (node->value.type.variable_type != VariableType::COL_SIMPLEX_MATRIX and
      node->value.type != AtomicType::PROBABILITY and
      node->value.type != AtomicType::REAL and
      node->value.type != AtomicType::POS_REAL and
      node->value.type != AtomicType::BOOLEAN) {
    return "NMC only supported on bool/probability/real/positive nodes -- failing on node " +
        std::to_string(node->index);
  } else {
    return "";
  }
}

void Graph::nmc(uint num_samples, uint seed, InferConfig infer_config) {
  NMC(this, seed).infer(num_samples, infer_config);
}

} // namespace graph
} // namespace beanmachine
