/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <vector>
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/stepper/single_site/single_site_stepping_method.h"
#include "beanmachine/graph/stepper/stepper.h"

namespace beanmachine {
namespace graph {

class MH;

class SequentialSingleSiteStepper : public Stepper {
 public:
  SequentialSingleSiteStepper(
      MH* mh,
      std::vector<SingleSiteSteppingMethod*> single_site_stepping_methods);

  void step() override;

  virtual ~SequentialSingleSiteStepper() override;

 protected:
  std::vector<SingleSiteSteppingMethod*> single_site_stepping_methods;

  std::vector<Stepper*> steppers;

  MH* mh;

  std::vector<Stepper*>& get_steppers();

  void make_steppers();

  SingleSiteSteppingMethod* find_applicable_single_site_stepping_method(
      Node* tgt_node);
};

} // namespace graph
} // namespace beanmachine
