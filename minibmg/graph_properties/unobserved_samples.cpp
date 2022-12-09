/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_properties/unobserved_samples.h"
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <unordered_set>

namespace {

using namespace beanmachine::minibmg;

class unobserved_samples_property
    : public Property<unobserved_samples_property, Graph, std::vector<Nodep>> {
 public:
  std::vector<Nodep>* create(const Graph& g) const override {
    auto result = new std::vector<Nodep>{};
    std::unordered_set<Nodep> observed_samples;
    for (auto& p : g.observations) {
      observed_samples.insert(p.first);
    }
    for (auto& node : g) {
      if (std::dynamic_pointer_cast<const ScalarSampleNode>(node) &&
          !observed_samples.contains(node)) {
        result->push_back(node);
      }
    }
    return result;
  }
};

} // namespace

namespace beanmachine::minibmg {

const std::vector<Nodep>& unobserved_samples(const Graph& graph) {
  return unobserved_samples_property::get(graph);
}

} // namespace beanmachine::minibmg
