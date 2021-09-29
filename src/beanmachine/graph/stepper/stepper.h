// Copyright (c) Facebook, Inc. and its affiliates.
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
