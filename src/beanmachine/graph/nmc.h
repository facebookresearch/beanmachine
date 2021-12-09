/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/mh.h"

namespace beanmachine {
namespace graph {

class NMC : public MH {
 public:
  virtual ~NMC();

  NMC(Graph* graph, uint seed);

  virtual std::string is_not_supported(Node* node) override;
};

} // namespace graph
} // namespace beanmachine
