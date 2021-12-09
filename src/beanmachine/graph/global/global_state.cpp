/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/global/global_state.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

GlobalState::GlobalState(Graph& g) : graph(g) {
  flat_size = 0;
  std::set<uint> supp = graph.compute_support();
  for (uint node_id : supp) {
    ordered_support.push_back(graph.nodes[node_id].get());
  }

  // initialize unconstrained value types
  // TODO: rename to initialize_unconstrained_value_types
  for (auto node : ordered_support) {
    if (node->is_stochastic() and node->node_type == NodeType::OPERATOR) {
      auto sto_node = static_cast<oper::StochasticOperator*>(node);
      sto_node->get_unconstrained_value(true);
    }
  }

  // save stochastic and deterministic nodes
  for (auto node : ordered_support) {
    if (node->is_stochastic() and !node->is_observed) {
      stochastic_nodes.push_back(node);
      // initialize vals_backup and grads_backup to correct size
      auto stochastic_node = static_cast<oper::StochasticOperator*>(node);
      NodeValue unconstrained_value =
          *stochastic_node->get_unconstrained_value(false);
      stochastic_unconstrained_vals_backup.push_back(unconstrained_value);
      stochastic_unconstrained_grads_backup.push_back(
          stochastic_node->back_grad1);
    } else if (!node->is_stochastic()) {
      deterministic_nodes.push_back(node);
    }
  }

  // calculate total size of unobserved unconstrained stochastic values
  for (Node* node : stochastic_nodes) {
    auto stochastic_node = static_cast<oper::StochasticOperator*>(node);
    NodeValue unconstrained_value =
        *stochastic_node->get_unconstrained_value(false);
    if (unconstrained_value.type.variable_type == VariableType::SCALAR) {
      flat_size++;
    } else {
      flat_size += static_cast<int>(unconstrained_value._matrix.size());
    }
  }
}

void GlobalState::initialize_values(InitType init_type, uint seed) {
  std::mt19937 gen(31 * seed + 17);
  if (init_type == InitType::PRIOR) {
    // Sample from stochastic nodes and update values directly
    for (auto node : stochastic_nodes) {
      auto sto_node = static_cast<oper::StochasticOperator*>(node);
      sto_node->eval(gen);
      sto_node->get_unconstrained_value(true); // TODO: rename this function
    }
  } else {
    // Update using set_flattened_unconstrained_values
    Eigen::VectorXd flattened_values(flat_size);
    if (init_type == InitType::RANDOM) {
      std::uniform_real_distribution<> uniform_real_distribution(-2, 2);
      for (int i = 0; i < flat_size; i++) {
        flattened_values[i] = uniform_real_distribution(gen);
      }
    } else if (init_type == InitType::ZERO) {
      flattened_values = Eigen::VectorXd::Zero(flat_size);
    }
    set_flattened_unconstrained_values(flattened_values);
  }

  // update and backup values, gradients, and log_prob
  update_backgrad();
  backup_unconstrained_values();
  backup_unconstrained_grads();
  update_log_prob();
}

void GlobalState::backup_unconstrained_values() {
  for (uint sto_node_id = 0;
       sto_node_id < static_cast<uint>(stochastic_nodes.size());
       sto_node_id++) {
    auto stochastic_node =
        static_cast<oper::StochasticOperator*>(stochastic_nodes[sto_node_id]);
    stochastic_unconstrained_vals_backup[sto_node_id] =
        *stochastic_node->get_unconstrained_value(false);
  }
}

void GlobalState::backup_unconstrained_grads() {
  for (uint sto_node_id = 0;
       sto_node_id < static_cast<uint>(stochastic_nodes.size());
       sto_node_id++) {
    stochastic_unconstrained_grads_backup[sto_node_id] =
        stochastic_nodes[sto_node_id]->back_grad1;
  }
}

void GlobalState::revert_unconstrained_values() {
  for (uint sto_node_id = 0;
       sto_node_id < static_cast<uint>(stochastic_nodes.size());
       sto_node_id++) {
    auto stochastic_node =
        static_cast<oper::StochasticOperator*>(stochastic_nodes[sto_node_id]);
    NodeValue* value = stochastic_node->get_unconstrained_value(false);
    *value = stochastic_unconstrained_vals_backup[sto_node_id];
    stochastic_node->get_original_value(true);
  }
}

void GlobalState::revert_unconstrained_grads() {
  for (uint sto_node_id = 0;
       sto_node_id < static_cast<uint>(stochastic_nodes.size());
       sto_node_id++) {
    stochastic_nodes[sto_node_id]->back_grad1 =
        stochastic_unconstrained_grads_backup[sto_node_id];
  }
}

void GlobalState::add_to_stochastic_unconstrained_nodes(
    Eigen::VectorXd& increment) {
  if (increment.size() != flat_size) {
    throw std::invalid_argument(
        "The size of increment is inconsistent with the values in the graph");
  }
  Eigen::VectorXd flattened_values;
  get_flattened_unconstrained_values(flattened_values);
  Eigen::VectorXd sum = flattened_values + increment;
  set_flattened_unconstrained_values(sum);
}

void GlobalState::get_flattened_unconstrained_values(
    Eigen::VectorXd& flattened_values) {
  flattened_values.resize(flat_size);
  int i = 0;
  for (Node* node : stochastic_nodes) {
    auto sto_node = static_cast<oper::StochasticOperator*>(node);
    NodeValue* value = sto_node->get_unconstrained_value(false);
    if (value->type.variable_type == VariableType::SCALAR) {
      flattened_values[i] = value->_double;
      i++;
    } else {
      Eigen::VectorXd vector(Eigen::Map<Eigen::VectorXd>(
          value->_matrix.data(), value->_matrix.size()));
      flattened_values.segment(i, vector.size()) = vector;
      i += static_cast<int>(value->_matrix.size());
    }
  }
}

void GlobalState::set_flattened_unconstrained_values(
    Eigen::VectorXd& flattened_values) {
  if (flattened_values.size() != flat_size) {
    throw std::invalid_argument(
        "The size of flattened_values is inconsistent with the values in the graph");
  }

  int i = 0;
  for (Node* node : stochastic_nodes) {
    // set unconstrained value
    auto sto_node = static_cast<oper::StochasticOperator*>(node);
    NodeValue* value = sto_node->get_unconstrained_value(false);
    if (value->type.variable_type == VariableType::SCALAR) {
      value->_double = flattened_values[i];
      i++;
    } else {
      value->_matrix = flattened_values.segment(i, value->_matrix.size());
      i += static_cast<int>(value->_matrix.size());
    }

    // sync value with unconstrained_value
    if (sto_node->transform_type != TransformType::NONE) {
      sto_node->get_original_value(true);
    }
  }
}

void GlobalState::get_flattened_unconstrained_grads(
    Eigen::VectorXd& flattened_grad) {
  flattened_grad.resize(flat_size);
  int i = 0;
  for (Node* node : stochastic_nodes) {
    if (node->value.type.variable_type == VariableType::SCALAR) {
      flattened_grad[i] = node->back_grad1._double;
      i++;
    } else {
      Eigen::VectorXd vector(Eigen::Map<Eigen::VectorXd>(
          node->back_grad1._matrix.data(), node->back_grad1._matrix.size()));
      flattened_grad.segment(i, vector.size()) = vector;
      i += static_cast<int>(node->back_grad1._matrix.size());
    }
  }
}

double GlobalState::get_log_prob() {
  return log_prob;
}

void GlobalState::update_log_prob() {
  log_prob = graph._full_log_prob(ordered_support);
}

void GlobalState::update_backgrad() {
  graph.update_backgrad(ordered_support);
}

} // namespace graph
} // namespace beanmachine
