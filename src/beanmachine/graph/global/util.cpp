/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/util.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/stochasticop.h"

namespace beanmachine {
namespace graph {

void set_default_transforms(Graph& g) {
  // add default transforms based on constraints
  // to transform all variables to the unconstrained space
  // POS_REAL variables -> LOG transform
  // TODO: add simplex transform
  for (uint node_id : g.compute_support()) {
    // @lint-ignore CLANGTIDY
    auto node = g.nodes[node_id].get();
    if (node->is_stochastic() and !node->is_observed) {
      auto sto_node = static_cast<oper::StochasticOperator*>(node);
      if (sto_node->transform_type == TransformType::NONE) {
        if (node->value.type.atomic_type == AtomicType::POS_REAL) {
          g.customize_transformation(TransformType::LOG, {node_id});
          // initialize the type of the unconstrained value
          // TODO: rename method to be more clear
          sto_node->get_unconstrained_value(true);
        } else if (node->value.type.atomic_type != AtomicType::REAL) {
          throw std::runtime_error(
              "Node " + std::to_string(node_id) +
              "cannot be automatically transformed to the unconstrained space");
        }
      }
    }
  }
}

} // namespace graph
} // namespace beanmachine
