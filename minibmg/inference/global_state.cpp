/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/inference/global_state.h"
#include <math.h>
#include <memory>
#include "beanmachine/graph/global/global_state.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/graph_properties/unobserved_samples.h"

namespace beanmachine::minibmg {

MinibmgGlobalState::MinibmgGlobalState(beanmachine::minibmg::Graph& graph)
    : graph{graph}, world{hmc_world_0(graph)} {
  samples.clear();
  // Since we only support scalars, we count the unobserved samples by ones.
  int num_unobserved_samples = -graph.observations.size();
  for (auto& node : graph) {
    if (std::dynamic_pointer_cast<const ScalarSampleNode>(node)) {
      num_unobserved_samples++;
    }
  }
  flat_size = num_unobserved_samples;
}

void MinibmgGlobalState::initialize_values(
    beanmachine::graph::InitType init_type,
    uint seed) {
  std::mt19937 gen(31 * seed + 17);
  std::vector<double>& samples = unconstrained_values;
  switch (init_type) {
    case graph::InitType::PRIOR: {
      // Evaluate the graph, saving samples.
      auto read_variable = [](const std::string&, const unsigned) -> Real {
        // there are no variables, so we don't have to read them.
        throw std::logic_error("models do not contain variables");
      };
      auto my_sampler = [&samples](
                            const Distribution<Real>& distribution,
                            std::mt19937& gen) -> SampledValue<Real> {
        auto result = sample_from_distribution(distribution, gen);
        // save the proposed value
        samples.push_back(result.unconstrained.as_double());
        return result;
      };
      auto eval_result = eval_graph<Real>(
          graph,
          gen,
          /* read_variable= */ read_variable,
          real_eval_data,
          /* run_queries= */ false,
          /* eval_log_prob= */ true,
          /* sampler = */ my_sampler);
    } break;
    case graph::InitType::RANDOM: {
      std::uniform_real_distribution<> uniform_real_distribution(-2, 2);
      for (int i = 0; i < flat_size; i++) {
        samples.push_back(uniform_real_distribution(gen));
      }
    } break;
    default: {
      for (int i = 0; i < flat_size; i++) {
        samples.push_back(0);
      }
    } break;
  }

  // update and backup values, gradients, and log_prob
  update_log_prob();
  update_backgrad();
  backup_unconstrained_values();
  backup_unconstrained_grads();
}

void MinibmgGlobalState::backup_unconstrained_values() {
  saved_unconstrained_values = unconstrained_values;
}

void MinibmgGlobalState::backup_unconstrained_grads() {
  saved_unconstrained_grads = unconstrained_grads;
}

void MinibmgGlobalState::revert_unconstrained_values() {
  unconstrained_values = saved_unconstrained_values;
}

void MinibmgGlobalState::revert_unconstrained_grads() {
  unconstrained_grads = saved_unconstrained_grads;
}

void MinibmgGlobalState::add_to_stochastic_unconstrained_nodes(
    Eigen::VectorXd& increment) {
  if (increment.size() != flat_size) {
    throw std::invalid_argument(
        "The size of increment is inconsistent with the values in the graph");
  }
  for (int i = 0; i < flat_size; i++) {
    unconstrained_values[i] += increment[i];
  }
}

void MinibmgGlobalState::get_flattened_unconstrained_values(
    Eigen::VectorXd& flattened_values) {
  flattened_values.resize(flat_size);
  for (int i = 0; i < flat_size; i++) {
    flattened_values[i] = unconstrained_values[i];
  }
}

void MinibmgGlobalState::set_flattened_unconstrained_values(
    Eigen::VectorXd& flattened_values) {
  if (flattened_values.size() != flat_size) {
    throw std::invalid_argument(
        "The size of flattened_values is inconsistent with the values in the graph");
  }
  for (int i = 0; i < flat_size; i++) {
    unconstrained_values[i] = flattened_values[i];
  }
}

void MinibmgGlobalState::get_flattened_unconstrained_grads(
    Eigen::VectorXd& flattened_grad) {
  flattened_grad.resize(flat_size);
  for (int i = 0; i < flat_size; i++) {
    flattened_grad[i] = unconstrained_grads[i];
  }
}

double MinibmgGlobalState::get_log_prob() {
  return log_prob;
}

void MinibmgGlobalState::update_log_prob() {
  log_prob = world->log_prob(this->unconstrained_values);
}

void MinibmgGlobalState::update_backgrad() {
  unconstrained_grads = world->gradients(this->unconstrained_values);
}

void MinibmgGlobalState::collect_sample() {
  auto queries = world->queries(this->unconstrained_values);
  std::vector<beanmachine::graph::NodeValue> compat_query;
  for (auto v : queries) {
    compat_query.emplace_back(v);
  }
  this->samples.emplace_back(compat_query);
}

std::vector<std::vector<beanmachine::graph::NodeValue>>&
MinibmgGlobalState::get_samples() {
  return samples;
}

void MinibmgGlobalState::set_default_transforms() {
  // minibmg always uses the default transforms
}

void MinibmgGlobalState::set_agg_type(
    beanmachine::graph::AggregationType agg_type) {
  if (agg_type != beanmachine::graph::AggregationType::NONE) {
    throw std::logic_error("unimplemented AggregationType");
  }
}

void MinibmgGlobalState::clear_samples() {
  samples.clear();
}

} // namespace beanmachine::minibmg
