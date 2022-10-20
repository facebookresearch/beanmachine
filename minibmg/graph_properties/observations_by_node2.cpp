/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_properties/observations_by_node2.h"

namespace {

using namespace beanmachine::minibmg;

class ObservationByNodeProperty
    : public Property<
          ObservationByNodeProperty,
          Graph2,
          const std::unordered_map<Node2p, double>> {
 public:
  const std::unordered_map<Node2p, double>* create(
      const Graph2& g) const override {
    return new std::unordered_map<Node2p, double>{
        g.observations.begin(), g.observations.end()};
  }
  ~ObservationByNodeProperty() {}
};

} // namespace

namespace beanmachine::minibmg {

const std::unordered_map<Node2p, double>& observations_by_node(
    const Graph2& graph) {
  return *ObservationByNodeProperty::get(graph);
}

} // namespace beanmachine::minibmg
