/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/mh.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/single_site_stepping_method.h"

namespace beanmachine {
namespace graph {

/*
 * An abstract default implementation of MH single-site stepping method
 * implementing the typical MH stepping method.
 * It uses a proposal provided by method get_proposal_distribution,
 * whose implementation is left to sub-classes.
 * Sub-classes must also implement methods is_applicable_to and
 * get_step_profiler_event.
 */
class DefaultSingleSiteSteppingMethod : public SingleSiteSteppingMethod {
 public:
  explicit DefaultSingleSiteSteppingMethod(MH* mh)
      : SingleSiteSteppingMethod(mh) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) override = 0;

  virtual void step(graph::Node* tgt_node) override;

 protected:
  virtual std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node) = 0;

  virtual ProfilerEvent get_step_profiler_event() = 0;
};

} // namespace graph
} // namespace beanmachine
