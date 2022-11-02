/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <stdexcept>

#include "beanmachine/graph/graph.h"

namespace beanmachine::graph {

// Marginalizes a discrete variable away from a given graph (in-place)
// (see case limitations below).
//
// Let us define *child* of a node 'n' as a descendant of 'n'
// that is a sample operator and whose path from 'n' includes
// deterministic nodes only (including distribution nodes).
//
// Example: consider the graph for the following model:
// n = sample(Normal(0,1))
// twice = n*2
// c1 = sample(Normal(n, 1.0))
// c2 = sample(Normal(twice, 1.0))
// c3 = sample(Normal(c2, 1.0))
//
// In such a graph, both c1 and c2 are
// children of n but c3 is not, even though
// it is a descendant of n (because the path
// from n to n3 involves other sample nodes).
//
// Let the initial graph have a discrete sample node d with distribution P(d),
// and set of values v1 and v2,
// let 'child' be the *only* child of d as defined above,
// let det_affected are the
// intervening deterministic nodes between d and 'child'
// (including 'child''s distribution),
// as in the following diagram:
//
//    Ancestors of P(d)
//           |
//           v
//          P(d) (discrete distribution over v1, v2)
//           |
//           v
//           d   (discrete sample node)
//           |
//           |  Other inputs
//           |       |
//           |       |
//           v       v
//         det_affected (deterministic nodes,
//           |          including distribution of 'child')
//           |
//           v
//         child (sole sample descendant of d)
//           |
//           v
//          ...
//
//  We wish the rewrite the graph to the following configuration,
//  where P(d=vi) is the prob of P(d) for value vi,
//  det_affected(d=vi) is a clone of set det_affected
//  where any inputs from d are replaced by the constant vi,
//  and the bimixture distribution selected among
//  det_affected(d=vi) according to weight_nodes P(d=vi).
//
//   Ancestors of P(d)               Other inputs
//         |                              |
//     +---+---+               +----------+--------+
//     |       |               |                   |
//     v       v               v                   v
//  P(d=v1) P(d=v2)    det_affected(d=v1)  det_affected(d=v2)
//     |       |               |                   |
//     v       v               v                   v
//  +-------------------------------------------------------+
//  |                 MIXTURE DISTRIBUTION                  |
//  +-------------------------------------------------------+
//                            |
//                            v
//                          child
//                            |
//                            v
//                           ...
//
// It holds that the final graph defines a joint probability
// distribution that is equal to the result of marginalizing d
// from the original joint probability distribution.
//
// Sampling from the resulting graph should produce better samples
// since each sample follows from information about both values of d.
//
// Current limitations: method only works for d a Bernoulli variable
// with a single child.
//
void marginalize(const Node* discrete, Graph& graph);

/* Indicates whether a node is marginalizable from the graph. */
bool is_marginalizable(Graph& graph, const Node* node);

void marginalize_all_marginalizable_variables(Graph& graph);

/**** Exceptions ****/

struct marginalization_on_non_sample_node : std::invalid_argument {
  marginalization_on_non_sample_node()
      : std::invalid_argument("Marginalization requested on non-sample node") {}
};

struct marginalization_on_observed_or_queried_node : std::invalid_argument {
  explicit marginalization_on_observed_or_queried_node()
      : std::invalid_argument(
            "Marginalization requested on observed or queried node") {}
};

struct marginalization_not_supported_for_samples_of : std::invalid_argument {
  explicit marginalization_not_supported_for_samples_of(
      DistributionType dist_type)
      : std::invalid_argument(
            std::string("Marginalization not yet supported for samples of ") +
            std::string(NAMEOF_ENUM(dist_type))) {}
};

struct marginalization_no_children : std::invalid_argument {
  marginalization_no_children()
      : std::invalid_argument(
            "Marginalization requested on discrete variable with no stochastic children") {
  }
};

struct marginalization_multiple_children : std::invalid_argument {
  marginalization_multiple_children()
      : std::invalid_argument(
            "Marginalization requested on discrete variable with multiple children") {
  }
};

} // namespace beanmachine::graph
