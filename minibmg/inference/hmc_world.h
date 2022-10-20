/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <memory>
#include <vector>
#include "beanmachine/minibmg/graph2.h"

namespace beanmachine::minibmg {

// The result of calling HMCWorld::evaluate
struct HMCWorldEvalResult;

// This interface defines an abstraction of a graph containing exactly and only
// what is needed for some inference methods, such as gradient ascent
// evaluation of the maximum likelihood, HMC, and NUTS.  The premise of this API
// is that the model being abstracted contains no unobserved samples from
// discrete distributions.  The distributions are assumed to be
// supported over the full range of real numbers; the implementation of this
// interface is expected to ensure this by transforming the distributions and
// their samples if necessary.
class HMCWorld {
 public:
  // The total number of samples appearing in the model that are NOT observed.
  virtual unsigned num_unobserved_samples() const = 0;

  // Given proposed assigned values for all of the onobserved samples in the
  // model (transformed, if necessary, so that they are unconstrained -
  // supported over the real numbers), given in the same order as the
  // observation nodes appear in the graph, compute the log probability of the
  // model with that assignment, as well as the the first derivative of the log
  // probability with respect to each of the proposed values.  The input vector
  // is required to be of a size that is equal to the return value of
  // num_unobserved_samples.
  virtual HMCWorldEvalResult evaluate(
      const std::vector<double>& proposed_unconstrained_values) const = 0;

  // Given proposed assigned values for all of the onobserved samples in the
  // model (as in evaluate), compute the value of queried nodes in the model.
  // Queried values are returned in the untransformed space.  The input
  // vector is required to be of a size that is equal to the return value of
  // num_unobserved_samples.
  virtual std::vector<double> queries(
      const std::vector<double>& proposed_unconstrained_values) const = 0;

  virtual ~HMCWorld() {}
};

struct HMCWorldEvalResult {
  // The computed log probability of a given assignment to the samples in a
  // model.
  double log_prob;

  // The derivative of the log probability with respect to each of the
  // unobserved samples in a model.
  std::vector<double> gradients;
};

// produce an abstraction of the graph for use by inference.  This
// implementation performs its work by evaluating the graph node by node on
// demand.  You can think of this as an interpreter for the graph.
std::unique_ptr<const HMCWorld> hmc_world_0(const Graph2& graph);

// produce an abstraction of the graph for use by inference.  This
// implementation performs its work by evaluating the graph symbolically,
// optimizing the symbolic form, and cacheing it for later use.  Then it
// evaluates the optimized expression graph when needed.  You can think of this
// as a bytecode compiler and bytecode interpreter for the graph.
std::unique_ptr<const HMCWorld> hmc_world_1(const Graph2& graph);

// produce an abstraction of the graph for use by inference.  This
// implementation performs its work by evaluating the graph symbolically,
// optimizing the symbolic form, generating native machine code from that, and
// cacheing that code for later use.  Then it calls the generated native code
// when needed.  This can be considered a JIT compiler for the graph.
std::unique_ptr<const HMCWorld> hmc_world_2(const Graph2& graph);

} // namespace beanmachine::minibmg
