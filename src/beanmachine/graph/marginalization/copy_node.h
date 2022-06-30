/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <string>
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class CopyNode : public Node {
 public:
  bool is_stochastic() const override {
    return false;
  }
  bool needs_gradient() const override {
    return true;
  }
  explicit CopyNode(Node* node_to_copy);
  void eval(std::mt19937& gen) override;
  double log_prob() const override;
  virtual ~CopyNode() override {}

 private:
  Node* node_to_copy;
};
} // namespace graph
} // namespace beanmachine
