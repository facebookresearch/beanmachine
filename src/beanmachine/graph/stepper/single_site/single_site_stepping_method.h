/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class MH;

// An abstraction for code taking a single-site MH step.
class SingleSiteSteppingMethod {
 public:
  explicit SingleSiteSteppingMethod(MH* mh) : mh(mh) {}

  virtual bool is_applicable_to(graph::Node* tgt_node) = 0;

  virtual void step(graph::Node* tgt_node) = 0;

  virtual ~SingleSiteSteppingMethod() {}

 protected:
  MH* mh;
};

} // namespace graph
} // namespace beanmachine
