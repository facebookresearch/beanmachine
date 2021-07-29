// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/nmc_single_site_stepper.h"

namespace beanmachine {
namespace graph {

class NMCScalarSingleSiteStepper : public NMCSingleSiteStepper {
 public:
  NMCScalarSingleSiteStepper(Graph* graph, NMC* nmc)
      : NMCSingleSiteStepper(graph, nmc) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) override;

  virtual void step(
      graph::Node* tgt_node,
      const std::vector<graph::Node*>& det_affected_nodes,
      const std::vector<graph::Node*>& sto_affected_nodes) override;

 private:
  std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node,
      NodeValue value,
      const std::vector<Node*>& det_affected_nodes,
      const std::vector<Node*>& sto_affected_nodes);
};

} // namespace graph
} // namespace beanmachine
