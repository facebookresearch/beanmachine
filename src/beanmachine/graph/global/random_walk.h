/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class RandomWalkMH : public GlobalMH {
 public:
  RandomWalkMH(Graph& g, double step_size);
};

} // namespace graph
} // namespace beanmachine
