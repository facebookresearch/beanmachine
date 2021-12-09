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

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/nmc.h"
#include "beanmachine/graph/stepper/single_site/nmc_dirichlet_beta_single_site_stepping_method.h"
#include "beanmachine/graph/stepper/single_site/nmc_dirichlet_gamma_single_site_stepping_method.h"
#include "beanmachine/graph/stepper/single_site/nmc_scalar_single_site_stepping_method.h"
#include "beanmachine/graph/stepper/single_site/sequential_single_site_stepper.h"

namespace beanmachine {
namespace graph {

// A stepper implementing NMC; this is separate from NMC algorithm so that it
// can be easily combined with other steppers.
class NMCStepper : public SequentialSingleSiteStepper {
 public:
  explicit NMCStepper(MH* mh)
      : SequentialSingleSiteStepper(
            mh,
            std::vector<SingleSiteSteppingMethod*>{
                // Note: the order of steppers below is important
                // because DirichletGamma is also applicable to
                // nodes to which Beta is applicable,
                // but we want to give priority to Beta in those cases.
                new NMCScalarSingleSiteSteppingMethod(mh),
                new NMCDirichletBetaSingleSiteSteppingMethod(mh),
                new NMCDirichletGammaSingleSiteSteppingMethod(mh)}) {}
};

} // namespace graph
} // namespace beanmachine
