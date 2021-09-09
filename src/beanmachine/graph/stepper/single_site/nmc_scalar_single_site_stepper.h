// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/default_single_site_stepper.h"

namespace beanmachine {
namespace graph {

class NMCScalarSingleSiteStepper : public DefaultSingleSiteStepper {
 public:
  NMCScalarSingleSiteStepper(Graph* graph, MH* mh)
      : DefaultSingleSiteStepper(graph, mh) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) override;

 protected:
  virtual std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node) override;

  virtual ProfilerEvent get_step_profiler_event() override;
};

} // namespace graph
} // namespace beanmachine
