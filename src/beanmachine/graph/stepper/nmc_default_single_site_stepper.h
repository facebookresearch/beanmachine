// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/nmc_single_site_stepper.h"

namespace beanmachine {
namespace graph {

/*
 * An abstract default implementation of NMC single-site stepper
 * implementing the typical MH step.
 * It uses a proposal provided by method get_proposal_distribution,
 * whose implementation is left to sub-classes.
 * Sub-classes must also implement methods is_applicable_to and
 * get_step_profiler_event.
 */
class NMCDefaultSingleSiteStepper : public NMCSingleSiteStepper {
 public:
  NMCDefaultSingleSiteStepper(Graph* graph, NMC* nmc)
      : NMCSingleSiteStepper(graph, nmc) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) override = 0;

  virtual void step(
      graph::Node* tgt_node,
      const std::vector<graph::Node*>& det_affected_nodes,
      const std::vector<graph::Node*>& sto_affected_nodes) override;

 protected:
  virtual std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node,
      const std::vector<Node*>& det_affected_nodes,
      const std::vector<Node*>& sto_affected_nodes) = 0;

  virtual ProfilerEvent get_step_profiler_event() = 0;
};

} // namespace graph
} // namespace beanmachine
