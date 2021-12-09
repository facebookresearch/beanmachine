/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <beanmachine/graph/distribution/distribution.h>
#include <beanmachine/graph/graph.h>
#include <beanmachine/graph/util.h>
#include <cmath>
#include <unordered_set>

namespace beanmachine {
namespace graph {

void Graph::cavi(
    uint num_iters,
    uint steps_per_iter,
    std::mt19937& gen,
    uint elbo_samples) {
  // convert the smart pointers in nodes to dumb pointers in node_ptrs
  // for faster access
  std::vector<Node*> node_ptrs;
  // store all the sampled values for each node
  std::vector<std::vector<NodeValue>> var_samples;
  for (uint node_id = 0; node_id < static_cast<uint>(nodes.size()); node_id++) {
    node_ptrs.push_back(nodes[node_id].get());
    var_samples.push_back(std::vector<NodeValue>());
  }
  assert(node_ptrs.size() > 0); // keep linter happy
  std::set<uint> supp = compute_support();
  // the variational parameter probability for each node (initially 0.5)
  std::vector<double> param_probability =
      std::vector<double>(nodes.size(), 0.5);
  assert(param_probability.size() > 0); // keep linter happy
  // compute pool : nodes that we will infer over
  //    -> nodes to sample, nodes to eval, nodes to log_prob
  // NOTE: we want the list of nodes in the pool to be sorted to ensure
  // that we update the nodes in topological order. This helps in some models
  // where some of the ancestor nodes have deterministic probabilities.
  std::map<
      uint,
      std::tuple<std::vector<uint>, std::vector<uint>, std::vector<uint>>>
      pool;
  for (uint node_id : supp) {
    Node* node = node_ptrs[node_id];
    if (not node->is_observed) {
      node->eval(gen); // evaluate the value of non-observed operator nodes
    }
    if (node->is_stochastic() and not node->is_observed) {
      // sample some values for this node
      auto& samples = var_samples[node_id];
      std::bernoulli_distribution distrib(param_probability[node_id]);
      for (uint step = 0; step < steps_per_iter; step++) {
        samples.push_back(NodeValue(bool(distrib(gen))));
      }
      // For each node in the pool we need its stochastic descendants
      // because those are the nodes for which we will compute the expected
      // log_prob. We will call these nodes the log_prob_nodes.
      std::vector<uint> det_desc;
      std::vector<uint> logprob_nodes;
      std::tie(det_desc, logprob_nodes) = compute_affected_nodes(node_id, supp);
      // In order to compute the log_prob of these nodes we need to
      // materialize their ancestors both deterministic and stochastic.
      // The unobserved stochastic ancestors are to be sampled while the
      // deterministic ancestors will be "eval"ed hence we call these nodes the
      // sample_nodes and the eval_nodes respectively. Note: the unobserved
      // logprob_nodes also need to be sampled excluding the target node.
      // To avoid duplicates and to sort the nodes we will first create sets.
      std::unordered_set<uint> sample_set;
      // the deterministic nodes have to be evaluated in topological order
      std::set<uint> det_set;
      for (auto id : logprob_nodes) {
        for (auto id2 : node_ptrs[id]->det_anc) {
          det_set.insert(id2);
        }
        for (auto id2 : node_ptrs[id]->sto_anc) {
          if (id2 != node_id and observed.find(id2) == observed.end()) {
            sample_set.insert(id2);
          }
        }
        if (id != node_id and observed.find(id) == observed.end()) {
          sample_set.insert(id);
        }
      }
      std::vector<uint> eval_nodes;
      eval_nodes.insert(eval_nodes.end(), det_set.begin(), det_set.end());
      std::vector<uint> sample_nodes;
      sample_nodes.insert(
          sample_nodes.end(), sample_set.begin(), sample_set.end());
      pool[node_id] = std::make_tuple(sample_nodes, eval_nodes, logprob_nodes);
    }
  }
  // optimization outer loop
  for (uint inum = 0; inum < num_iters; inum++) {
    for (auto it = pool.begin(); it != pool.end(); ++it) {
      uint tgt_node_id = it->first;
      Node* tgt_node = node_ptrs[tgt_node_id];
      // the following dance of getting into a tuple is needed because this
      // version of C++ doesn't have structured bindings
      std::tuple<
          const std::vector<uint>&,
          const std::vector<uint>&,
          const std::vector<uint>&>
          tmp_tuple = it->second;
      const std::vector<uint>& sample_nodes = std::get<0>(tmp_tuple);
      const std::vector<uint>& eval_nodes = std::get<1>(tmp_tuple);
      const std::vector<uint>& logprob_nodes = std::get<2>(tmp_tuple);
      std::vector<double> expec(2, 0.0);
      for (uint step = 0; step < steps_per_iter; step++) {
        for (uint node_id : sample_nodes) {
          node_ptrs[node_id]->value = var_samples[node_id][step];
        }
        for (uint val = 0; val < 2; val++) {
          tgt_node->value = NodeValue(bool(val));
          for (uint node_id : eval_nodes) {
            node_ptrs[node_id]->eval(gen);
          }
          double log_prob = 0;
          for (uint node_id : logprob_nodes) {
            log_prob += node_ptrs[node_id]->log_prob();
          }
          // update the expectation w.r.t. current value of target node
          expec[val] += log_prob / steps_per_iter;
        }
      }
      if (std::isfinite(expec[0]) or std::isfinite(expec[1])) {
        param_probability[tgt_node_id] = util::logistic(expec[1] - expec[0]);
      } else {
        param_probability[tgt_node_id] = 0.5;
      }
      auto& samples = var_samples[tgt_node_id];
      std::bernoulli_distribution distrib(param_probability[tgt_node_id]);
      for (uint step = 0; step < steps_per_iter; step++) {
        samples[step] = NodeValue(bool(distrib(gen)));
      }
    }
    if (elbo_samples > 0) {
      // For a model p(X, Z) assume we are trying to estimate p(Z | X=x)
      //  using a variational approximation Q(Z).
      // Now KL-Divergence of  Q(Z) || p(Z|x) >= 0
      // => E[log Q(Z) - log p(Z|x) | Z ~ Q] >= 0
      // => p(x) >= E[log p(Z, x) - log Q(Z) | Z ~ Q]
      // the RHS is the ELBO or evidence lower bound
      // We compute this expectation using the samples of the nodes in our pool.
      // log p(Z, x) is the log_prob of all the stochastic nodes
      // and log Q(Z) is the log of the variational distribution for the pool.
      double elbo = 0;
      for (uint step = 0; step < elbo_samples; step++) {
        for (uint node_id : supp) {
          Node* node = node_ptrs[node_id];
          if (node->is_stochastic()) {
            if (not node->is_observed) {
              double prob = param_probability[node_id];
              std::bernoulli_distribution distrib(prob);
              node->value = NodeValue(bool(distrib(gen)));
              // subtract the log_prob of the variational distribution
              elbo -= node->value._bool ? log(prob) : log(1 - prob);
            }
            // add the log_prob of the joint distribution
            elbo += node->log_prob();
          } else if (node->node_type == NodeType::OPERATOR) {
            node->eval(gen);
          }
        }
      }
      elbo_vals.push_back(elbo / elbo_samples);
    }
  }
  variational_params.clear();
  for (uint node_id : queries) {
    variational_params.push_back({param_probability[node_id]});
  }
}

} // namespace graph
} // namespace beanmachine
