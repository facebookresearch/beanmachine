// Copyright (c) Facebook, Inc. and its affiliates.
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/nmc.h"
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

NMC::NMC(Graph* g, uint seed)
    : MH(g,
         seed,
         // Note: the order of steppers below is important
         // because DirichletGamma is also applicable to
         // nodes to which Beta is applicable,
         // but we want to give priority to Beta in those cases.
         new SequentialSingleSiteStepper( // okay to allocate but not deallocate
                                          // because MH takes ownership of
                                          // stepper.
             g,
             this,
             std::vector<SingleSiteSteppingMethod*>{
                 new NMCScalarSingleSiteSteppingMethod(g, this),
                 new NMCDirichletBetaSingleSiteSteppingMethod(g, this),
                 new NMCDirichletGammaSingleSiteSteppingMethod(g, this)})) {}

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
