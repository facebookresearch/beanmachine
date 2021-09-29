// Copyright (c) Facebook, Inc. and its affiliates.
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
      Graph* graph,
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
