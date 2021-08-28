// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class NMC;

// An abstraction for code taking a single-site NMC step.
// TODO: we will eventually have a more abstract SingleSiteStepper
// that is not NMC-specific.
class NMCSingleSiteStepper {
 public:
  NMCSingleSiteStepper(Graph* graph, NMC* nmc) : graph(graph), nmc(nmc) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) = 0;

  virtual void step(graph::Node* tgt_node) = 0;

  virtual ~NMCSingleSiteStepper() {}

 protected:
  Graph* graph;
  NMC* nmc;
};

} // namespace graph
} // namespace beanmachine
