// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/nmc_single_site_stepper.h"

namespace beanmachine {
namespace graph {

class NMCDirichletGammaSingleSiteStepper : public NMCSingleSiteStepper {
 public:
  NMCDirichletGammaSingleSiteStepper(Graph* graph, NMC* nmc)
      : NMCSingleSiteStepper(graph, nmc) {}

  virtual void step(
      graph::Node* tgt_node,
      const std::vector<graph::Node*>& det_affected_nodes,
      const std::vector<graph::Node*>& sto_affected_nodes);

 private:
  std::unique_ptr<proposer::Proposer> create_proposer_dirichlet_gamma(
      const std::vector<Node*>& sto_nodes,
      Node* tgt_node,
      double param_a,
      NodeValue value,
      /* out */ double& logweight);
};

} // namespace graph
} // namespace beanmachine
