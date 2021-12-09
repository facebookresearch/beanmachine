/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/stepper/single_site/sequential_single_site_stepper.h"
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/mh.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/single_site_stepper.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

// Builds a sequential single site stepper which sequentially applies
// single-site steppers for each of MH's unobserved stochastic nodes
// based on given single-site stepping methods.
// IMPORTANT: takes ownership of single-site stepping methods, deleting them at
// destruction time.
SequentialSingleSiteStepper::SequentialSingleSiteStepper(
    MH* mh,
    std::vector<SingleSiteSteppingMethod*> single_site_stepping_methods)
    : Stepper(mh),
      single_site_stepping_methods(single_site_stepping_methods),
      mh(mh) {}

std::vector<Stepper*>& SequentialSingleSiteStepper::get_steppers() {
  if (static_cast<uint>(steppers.size()) == 0) {
    make_steppers();
  }
  return steppers;
}

void SequentialSingleSiteStepper::make_steppers() {
  for (uint i = 0;
       i < static_cast<uint>(mh->unobserved_stochastic_support().size());
       ++i) {
    auto tgt_node = mh->unobserved_stochastic_support()[i];
    auto single_site_stepping_method =
        find_applicable_single_site_stepping_method(tgt_node);
    steppers.push_back(
        new SingleSiteStepper(single_site_stepping_method, tgt_node, mh));
  }
}

SingleSiteSteppingMethod*
SequentialSingleSiteStepper::find_applicable_single_site_stepping_method(
    Node* tgt_node) {
  auto applicable_stepper = std::find_if(
      single_site_stepping_methods.begin(),
      single_site_stepping_methods.end(),
      [tgt_node](auto st) { return st->is_applicable_to(tgt_node); });

  if (applicable_stepper == single_site_stepping_methods.end()) {
    throw std::runtime_error(
        "No single-site stepping method applies to node " +
        std::to_string(tgt_node->index));
  }

  return *applicable_stepper;
}

void SequentialSingleSiteStepper::step() {
  for (auto stepper : get_steppers()) {
    stepper->step();
  }
}

SequentialSingleSiteStepper::~SequentialSingleSiteStepper() {
  for (auto stepper : get_steppers()) {
    delete stepper;
  }
  for (auto single_site_stepping_method : single_site_stepping_methods) {
    delete single_site_stepping_method;
  }
}

} // namespace graph
} // namespace beanmachine
