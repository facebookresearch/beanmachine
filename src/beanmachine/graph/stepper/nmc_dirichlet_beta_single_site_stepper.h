// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/nmc_single_site_stepper.h"

namespace beanmachine {
namespace graph {

class NMCDirichletBetaSingleSiteStepper : public NMCSingleSiteStepper {
 public:
  NMCDirichletBetaSingleSiteStepper(Graph* graph, NMC* nmc)
      : NMCSingleSiteStepper(graph, nmc) {}
  virtual bool is_applicable_to(graph::Node* tgt_node) override;

  virtual void step(
      graph::Node* tgt_node,
      const std::vector<graph::Node*>& det_affected_nodes,
      const std::vector<graph::Node*>& sto_affected_nodes) override;

 private:
  std::unique_ptr<proposer::Proposer> create_proposer_dirichlet_beta(
      const std::vector<Node*>& sto_nodes,
      Node* tgt_node,
      double param_a,
      double param_b,
      NodeValue value,
      /* out */ double& logweight);
};

} // namespace graph
} // namespace beanmachine
