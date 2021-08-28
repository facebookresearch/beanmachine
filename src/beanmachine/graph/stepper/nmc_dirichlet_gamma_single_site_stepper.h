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
  virtual bool is_applicable_to(graph::Node* tgt_node) override;

  virtual void step(graph::Node* tgt_node) override;

 private:
  std::unique_ptr<proposer::Proposer> create_proposal_dirichlet_gamma(
      Node* tgt_node,
      double param_a,
      NodeValue value,
      /* out */ double& logweight);
};

} // namespace graph
} // namespace beanmachine
