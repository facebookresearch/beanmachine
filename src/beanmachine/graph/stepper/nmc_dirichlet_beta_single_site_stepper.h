// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/nmc_default_single_site_stepper.h"

namespace beanmachine {
namespace graph {

class NMCDirichletBetaSingleSiteStepper : public NMCDefaultSingleSiteStepper {
 public:
  NMCDirichletBetaSingleSiteStepper(Graph* graph, NMC* nmc)
      : NMCDefaultSingleSiteStepper(graph, nmc) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) override;

 protected:
  virtual std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node) override;

  virtual ProfilerEvent get_step_profiler_event() override;
};

} // namespace graph
} // namespace beanmachine
