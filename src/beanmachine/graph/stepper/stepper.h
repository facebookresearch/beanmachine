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

// An abstraction for code taking a MH step.
class Stepper {
 public:
  explicit Stepper(MH* mh) : mh(mh) {}

  virtual void step() = 0;

  virtual ~Stepper() {}

 protected:
  MH* mh;
};

} // namespace graph
} // namespace beanmachine
