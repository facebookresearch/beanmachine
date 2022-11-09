/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/core.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/transform.hpp>
#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/marginalization/marginalization_extensional.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/third-party/nameof.h"
#include "beanmachine/graph/util.h"

namespace beanmachine::graph {

using namespace ranges;
using namespace std;
using namespace util;
using namespace distribution;
using namespace oper;

/*
Information about the child of the discrete marginalized variable.
child: the child if there is a single one, or nullptr otherwise.
number_of_children: how many children were found.
*/
struct ChildInfo {
  Node* child;
  size_t number_of_children;
};

ChildInfo get_child_info(const vector<Node*>& det_affected) {
  // clang-format off
  auto children = det_affected
                | views::filter(is(NodeType::DISTRIBUTION))
                | views::transform(get_out_nodes)
                | views::join // flattens sets of out-nodes
                | views::filter(is_sample_or_sample_iid)
                | to<vector>();
  // clang-format on

  auto child = children.size() == 1 ? children[0] : nullptr;
  return {child, children.size()};
}

/*
Information about the local descandants of a marginalized variable.
Contains the deterministic affected nodes, the child if there is a single one,
and the number of children found.
*/
struct LocalDescendants {
  LocalDescendants(vector<Node*> det_affected, ChildInfo child_info)
      : det_affected(det_affected),
        child(child_info.child),
        number_of_children(child_info.number_of_children) {}
  vector<Node*> det_affected;
  Node* child;
  size_t number_of_children;
};

LocalDescendants get_local_descendants(Graph& graph, const Node* discrete) {
  AffectedNodes affected_ids = graph.compute_affected_nodes_except_self(
      discrete->index, graph.compute_support());
  const auto& det_affected_ids = get<0>(affected_ids);
  auto det_affected = graph.convert_node_ids(det_affected_ids);
  return {det_affected, get_child_info(det_affected)};
}

bool discrete_is_a_sample_node(const Node* discrete) {
  return is_sample_or_sample_iid(discrete);
}

bool discrete_is_not_observed_or_queried(
    const Node* discrete,
    const Graph& graph) {
  return not(
      contains(graph.observed, discrete->index) or
      contains(graph.queries, discrete->index));
}

DistributionType distribution_type(const Node* discrete) {
  assert(discrete->in_nodes.size() == 1);
  auto parent = discrete->in_nodes[0];
  return dynamic_cast<Distribution*>(parent)->dist_type;
}

bool distribution_is_supported(const Node* discrete) {
  assert(discrete->in_nodes.size() == 1);
  auto parent = discrete->in_nodes[0];
  Bernoulli* bernoulli = dynamic_cast<Bernoulli*>(parent);
  return bernoulli != nullptr;
}

/* Indicates whether a node is marginalizable from the graph. */
bool is_marginalizable(Graph& graph, const Node* discrete) {
  return discrete_is_a_sample_node(discrete) and
      discrete_is_not_observed_or_queried(discrete, graph) and
      distribution_is_supported(discrete) and
      get_local_descendants(graph, discrete).number_of_children == 1;
}

void check_discrete_is_a_sample_node(const Node* discrete) {
  if (not discrete_is_a_sample_node(discrete)) {
    throw marginalization_on_non_sample_node();
  }
}

void check_discrete_is_not_observed_or_queried(
    const Node* discrete,
    const Graph& graph) {
  if (not discrete_is_not_observed_or_queried(discrete, graph)) {
    throw marginalization_on_observed_or_queried_node();
  }
}

void check_distribution_is_supported(const Node* discrete) {
  if (not distribution_is_supported(discrete)) {
    throw marginalization_not_supported_for_samples_of(
        distribution_type(discrete));
  }
}

/*
Returns local descendants if they contain a single child ("valid")
or throw an appropriate exception.
*/
LocalDescendants get_valid_local_descendants_or_throw_exceptions(
    Graph& graph,
    const Node* discrete) {
  auto local_descendants = get_local_descendants(graph, discrete);

  switch (local_descendants.number_of_children) {
    case 0:
      throw marginalization_no_children();
    case 1:
      return local_descendants;
    default:
      throw marginalization_multiple_children();
  }
}

vector<NodeValue> values_of(const Node* discrete) {
  // Only distribution supported as of yet is Bernoulli:
  return {NodeValue(false), NodeValue(true)};
}

const Distribution* distribution_of(const Node* node) {
  assert(node->in_nodes.size() == 1);
  return dynamic_cast<const Distribution*>(node->in_nodes[0]);
}

Node* add_mixture(
    Graph& graph,
    const vector<Node*>& weight_nodes,
    const vector<Node*>& distributions) {
  // weight w_i of the i-th is a log unnormalized probability
  // (because Distributions are only required to provide
  // log unnormalized probabilities).
  // That is to say,
  // w_i = log c p_i = log c + log p_i
  // for some constant c.
  // The Bimixture distribution requires the value p0 (probability
  // of the first value).
  // This means exp(w_i) is the unnormalized probability of i-th value.
  // Therefore, p0 results from normalizing the set of exp(w_i):
  // p0 = exp(w0) / sum_i exp(w_i).
  // Because of numeric reasons, we do not want to compute sum_i exp(w_i)
  // directly, but instead use the logsumexp(w) function
  // which computes log(sum_i exp(w_i)).
  // This can be done by computing instead log p0 and exponentiating it later:
  // p0 = exp(log p0) = exp(log(exp(w0) / sum_i exp(w_i)))
  // = exp( log(exp(w0)) - log(sum_i exp(w_i)) )
  // = exp( w0 - logsumexp(w) )
  // We assemble the nodes computing this expression:
  auto weight_ids = graph.get_node_ids(weight_nodes);
  assert(weight_ids.size() != 0);
  auto logsumexp_w = graph.add_operator(OperatorType::LOGSUMEXP, weight_ids);
  auto minus_one = graph.add_constant_real(-1);
  auto minus_logsumexp_w =
      graph.add_operator(OperatorType::MULTIPLY, {minus_one, logsumexp_w});
  auto log_prob =
      graph.add_operator(OperatorType::ADD, {weight_ids[0], minus_logsumexp_w});
  auto prob_real = graph.add_operator(OperatorType::EXP, {log_prob});
  auto prob = graph.add_operator(OperatorType::TO_PROBABILITY, {prob_real});
  Distribution* distr0 = dynamic_cast<Distribution*>(distributions[0]);
  auto mixture = graph.add_distribution(
      DistributionType::BIMIXTURE,
      distr0->sample_type,
      {prob, distributions[0]->index, distributions[1]->index});
  return graph.get_node(mixture);
}

void marginalize(const Node* discrete, Graph& graph) {
  check_discrete_is_a_sample_node(discrete);
  check_discrete_is_not_observed_or_queried(discrete, graph);
  check_distribution_is_supported(discrete);

  // Capture neighborhood of 'discrete':
  // The distribution of 'discrete':
  auto prior_discrete = distribution_of(discrete);
  auto valid_local_descendants = // guaranteed to contain a single child
      get_valid_local_descendants_or_throw_exceptions(graph, discrete);
  // the deterministic nodes immediately affected by discrete before
  // we find a stochastic variable:
  auto det_affected = valid_local_descendants.det_affected;
  // the (required to be the only one for now) sample node
  // descending from 'discrete' through deterministic nodes only:
  auto child = valid_local_descendants.child;

  // As we duplicate det_affected, the child's distribution will always
  // occupy the same position in the list of nodes.
  // We determine this position's index here to find these distributions
  // for each value later.
  auto child_dist = child->in_nodes[0];
  auto child_dist_index_in_det_affected = find_index(det_affected, child_dist);

  // Function making a constant node for a given 'discrete' value:
  auto make_constant_node = std::function([&](const NodeValue& v) {
    auto id = graph.add_constant(v);
    return graph.get_node(id);
  });

  // Function making a node for the weight of a given 'discrete' value,
  // equal to its log prob according to 'discrete''s distribution:
  auto make_weight_node = std::function([&](const Node* v) {
    auto id = graph.add_operator(
        OperatorType::LOG_PROB, {prior_discrete->index, v->index});
    return graph.get_node(id);
  });

  // Function making copy of det_affected for a given 'discrete' value:
  auto make_det_affected = std::function([&](Node* v) {
    auto duplicate_nodes = duplicate_subgraph(graph, det_affected);
    for (auto duplicate_node : duplicate_nodes) {
      graph.replace_edges(duplicate_node, discrete, v);
    }
    return duplicate_nodes;
  });

  // Function getting the child's distribution for a given 'discrete' value:
  auto get_child_dist =
      std::function([&](const vector<Node*>& value_det_affected) {
        return value_det_affected[child_dist_index_in_det_affected];
      });

  auto values = values_of(discrete);
  auto value_constant_nodes = map2vec(values, make_constant_node);
  auto value_weight_nodes = map2vec(value_constant_nodes, make_weight_node);
  auto value_det_affecteds = map2vec(value_constant_nodes, make_det_affected);
  auto value_child_distributions = map2vec(value_det_affecteds, get_child_dist);
  auto mixture_ptr =
      add_mixture(graph, value_weight_nodes, value_child_distributions);
  graph.set_edge(child, /* in-node index=*/0, /* new_in-node=*/mixture_ptr);
}

const Node* find_first_marginalizable(Graph& graph) {
  auto first = ::ranges::find_if(graph.node_ptrs(), [&](const Node* node) {
    return is_marginalizable(graph, node);
  });
  if (first != graph.node_ptrs().end()) {
    return *first;
  } else {
    return nullptr; // NOLINT(facebook-hte-NullableReturn)
  }
}

void marginalize_all_marginalizable_variables(Graph& graph) {
  while (auto next = find_first_marginalizable(graph)) {
    marginalize(next, graph);
  }
}

void Graph::automatic_discrete_marginalization(
    InferenceType base_inference_type,
    uint num_samples,
    uint seed,
    InferConfig infer_config) {
  auto graph_copy{*this}; // copy graph, since marginalization is done in-place.

  marginalize_all_marginalizable_variables(graph_copy);

  // automatic_discrete_marginalization is invoked by Graph::_infer
  // which in turn is invoked by Graph::infer.
  // Graph::infer initializes several Graph fields and _infer
  // stores samples in the appropriate fields
  // (depending on things like agg_type).
  // However, because we are running _infer on a *copy*
  // of the original graph, samples will be stored
  // in the appropriate fields in the graph copy.
  // Therefore, it is important to copy these fields
  // (which at this point have already been initialized)
  // to the graph copy, then run _infer to the copy
  // and, finally, copy back the results to the main graph.

  graph_copy.agg_type = agg_type;
  graph_copy.samples = samples;
  graph_copy.means = means;
  graph_copy.log_prob_vals = log_prob_vals;
  graph_copy.master_graph = master_graph;
  graph_copy.thread_index = thread_index;
  graph_copy.samples_allchains = samples_allchains;
  graph_copy.means_allchains = means_allchains;
  graph_copy.log_prob_allchains = log_prob_allchains;

  graph_copy._infer(num_samples, base_inference_type, seed, infer_config);

  agg_type = graph_copy.agg_type;
  samples = graph_copy.samples;
  means = graph_copy.means;
  log_prob_vals = graph_copy.log_prob_vals;
  // No need to copy parallel chains data since that
  // is already automatically placed in the master graph.
  // master_graph = graph_copy.master_graph;
  // thread_index = graph_copy.thread_index;
  // samples_allchains = graph_copy.samples_allchains;
  // means_allchains = graph_copy.means_allchains;
  // log_prob_allchains = graph_copy.log_prob_allchains;
}

} // namespace beanmachine::graph
