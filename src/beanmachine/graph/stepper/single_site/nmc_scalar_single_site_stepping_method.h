/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/default_single_site_stepping_method.h"

namespace beanmachine {
namespace graph {

class NMCScalarSingleSiteSteppingMethod
    : public DefaultSingleSiteSteppingMethod {
 public:
  explicit NMCScalarSingleSiteSteppingMethod(MH* mh)
      : DefaultSingleSiteSteppingMethod(mh) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) override;

 protected:
  virtual std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node) override;

  virtual ProfilerEvent get_step_profiler_event() override;
};

} // namespace graph
} // namespace beanmachine
