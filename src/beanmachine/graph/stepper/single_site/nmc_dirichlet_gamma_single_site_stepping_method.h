/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/single_site_stepping_method.h"

namespace beanmachine {
namespace graph {

class NMCDirichletGammaSingleSiteSteppingMethod
    : public SingleSiteSteppingMethod {
 public:
  explicit NMCDirichletGammaSingleSiteSteppingMethod(MH* mh)
      : SingleSiteSteppingMethod(mh) {}
  virtual bool is_applicable_to(graph::Node* tgt_node) override;

  virtual void step(graph::Node* tgt_node) override;

 private:
  double compute_sto_affected_nodes_log_prob(
      Node* tgt_node,
      double param_a,
      NodeValue value);
  std::unique_ptr<proposer::Proposer> create_proposal_dirichlet_gamma(
      Node* tgt_node,
      double param_a,
      double sum,
      NodeValue value,
      uint k);
};

} // namespace graph
} // namespace beanmachine
