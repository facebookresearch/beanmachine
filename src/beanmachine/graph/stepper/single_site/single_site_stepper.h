// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/stepper/single_site/single_site_stepping_method.h"
#include "beanmachine/graph/stepper/stepper.h"

namespace beanmachine {
namespace graph {

// A stepper based on a single-site stepping method bound to a specific node.
class SingleSiteStepper : public Stepper {
 public:
  SingleSiteStepper(
      SingleSiteSteppingMethod* single_site_stepping_method,
      Node* node,
      Graph* graph,
      MH* mh)
      : Stepper(graph, mh),
        single_site_stepping_method(single_site_stepping_method),
        node(node) {}

  void step() override {
    single_site_stepping_method->step(node);
  }

 protected:
  SingleSiteSteppingMethod* single_site_stepping_method;
  Node* node;
};

} // namespace graph
} // namespace beanmachine
