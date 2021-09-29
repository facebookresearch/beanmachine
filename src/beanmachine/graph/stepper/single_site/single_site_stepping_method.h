// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class MH;

// An abstraction for code taking a single-site MH step.
class SingleSiteSteppingMethod {
 public:
  SingleSiteSteppingMethod(Graph* graph, MH* mh) : graph(graph), mh(mh) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) = 0;

  virtual void step(graph::Node* tgt_node) = 0;

  virtual ~SingleSiteSteppingMethod() {}

 protected:
  Graph* graph;
  MH* mh;
};

} // namespace graph
} // namespace beanmachine
