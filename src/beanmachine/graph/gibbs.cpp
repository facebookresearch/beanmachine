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
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

void Graph::gibbs(uint num_samples, uint seed, InferConfig infer_config) {
  std::mt19937 gen(seed);
  std::set<uint> supp = compute_support();
  // eval each node so that we have a starting value and verify that these
  // values are all scalar
  // also compute the pool of variables that we will infer over and
  // compute their descendants -- i.e. all stochastic non-observed nodes
  // that are in the support of the graph
  // pool : nodes that we will infer over -> det_desc, sto_desc
  std::map<uint, std::tuple<std::vector<uint>, std::vector<uint>>> pool;
  // cache_logodds : nodes that we will infer over -> log odds of not changing
  std::vector<double> cache_logodds = std::vector<double>(nodes.size());
  assert(cache_logodds.size() > 0); // keep linter happy
  // inv_sto : stochastic node -> parent nodes in pool
  // x in sto_desc[y] => y in inv_sto[x]
  // this is a temp object which is needed to construct markov_blanket (below)
  std::map<uint, std::set<uint>> inv_sto;
  std::vector<Node*> ordered_supp;
  for (uint node_id : supp) {
    Node* node = nodes[node_id].get();
    bool node_is_not_observed = observed.find(node_id) == observed.end();
    if (node_is_not_observed) {
      node->eval(gen); // evaluate the value of non-observed operator nodes
    }
    if (node->is_stochastic() and node_is_not_observed) {
      std::vector<uint> det_nodes;
      std::vector<uint> sto_nodes;
      std::tie(det_nodes, sto_nodes) = compute_affected_nodes(node_id, supp);
      pool[node_id] = std::make_tuple(det_nodes, sto_nodes);
      cache_logodds[node_id] = NAN; // nan => needs to be re-computed
      for (auto sto : sto_nodes) {
        if (inv_sto.find(sto) == inv_sto.end()) {
          inv_sto[sto] = std::set<uint>();
        }
        inv_sto[sto].insert(node_id);
      }
    }
    if (infer_config.keep_log_prob) {
      ordered_supp.push_back(node);
    }
  }
  // markov_blanket of a node is the set of other nodes whose conditional
  // probability changes when the value of this node changes. This is a
  // symmetric relation and we only track it for the subset of nodes that
  // are in the pool of variables to be inferred over.
  // Formally, x in markov_blanket[y]
  //                <==> exists z s.t. z in sto_desc[x] and z in sto_desc[y]
  // Note: x is in markov_blanket[x]
  std::map<uint, std::set<uint>> markov_blanket;
  for (auto it = inv_sto.begin(); it != inv_sto.end(); ++it) {
    for (auto it1 = it->second.begin(); it1 != it->second.end(); ++it1) {
      if (markov_blanket.find(*it1) == markov_blanket.end()) {
        markov_blanket[*it1] = std::set<uint>();
      }
      for (auto it2 = it1; it2 != it->second.end(); ++it2) {
        markov_blanket[*it1].insert(*it2);
        if (markov_blanket.find(*it2) == markov_blanket.end()) {
          markov_blanket[*it2] = std::set<uint>();
        }
        markov_blanket[*it2].insert(*it1);
      }
    }
  }
  std::vector<NodeValue> old_values = std::vector<NodeValue>(nodes.size());
  assert(old_values.size() > 0); // keep linter happy
  // convert the smart pointers in nodes to dumb pointers in node_ptrs
  // for faster access
  std::vector<Node*> node_ptrs;
  for (uint node_id = 0; node_id < static_cast<uint>(nodes.size()); node_id++) {
    node_ptrs.push_back(nodes[node_id].get());
  }
  assert(node_ptrs.size() > 0); // keep linter happy
  // sampling outer loop
  for (uint snum = 0; snum < num_samples + infer_config.num_warmup; snum++) {
    for (auto it = pool.begin(); it != pool.end(); ++it) {
      bool must_change = false; // must_change => must change current value
      // if we have a cached value of the transition odds then use that instead
      if (not std::isnan(cache_logodds[it->first])) {
        // do we keep the current value?
        if (util::sample_logodds(gen, cache_logodds[it->first])) {
          continue;
        } else {
          must_change = true;
        }
      }
      // for the target sampled node grab its deterministic and stochastic
      // children
      // the following dance of getting into a tuple is needed because this
      // version of C++ doesn't have structured bindings
      std::tuple<const std::vector<uint>&, const std::vector<uint>&> tmp_tuple =
          it->second;
      const std::vector<uint>& det_nodes = std::get<0>(tmp_tuple);
      const std::vector<uint>& sto_nodes = std::get<1>(tmp_tuple);
      assert(it->first == sto_nodes.front());
      // now, compute the probability of all the stochastic nodes that are
      // going to be affected when we change the value of the target node
      double old_logweight = 0;
      for (uint node_id : sto_nodes) {
        const Node* node = node_ptrs[node_id];
        old_logweight += node->log_prob();
      }
      // save the values of the deterministic descendants of the target node
      // as well the target node itself
      for (uint node_id : det_nodes) {
        const Node* node = node_ptrs[node_id];
        old_values[node_id] = node->value;
      }
      Node* tgt_node = node_ptrs[it->first];
      old_values[it->first] = tgt_node->value;
      // propose a new value for the target node and update all the
      // deterministic children note: assuming only boolean values
      if (tgt_node->value.type != AtomicType::BOOLEAN) {
        throw std::runtime_error(
            "all stochastic random variables should be boolean");
      }
      tgt_node->value._bool = not tgt_node->value._bool; // flip
      for (uint node_id : det_nodes) {
        Node* node = node_ptrs[node_id];
        node->eval(gen);
      }
      // compute the probability of the stochastic nodes with the new value
      // of the target node
      double new_logweight = 0;
      for (uint node_id : sto_nodes) {
        const Node* node = node_ptrs[node_id];
        new_logweight += node->log_prob();
      }
      // compute logodds of keeping the current value
      double logodds = old_logweight - new_logweight;
      // Time to make a decision! Do we keep the old value or pick a new value.
      if ((not must_change) and util::sample_logodds(gen, logodds)) {
        // if the move to the new value is rejected then we need to restore
        // all the deterministic decendants and the target node to original
        // values
        for (uint node_id : det_nodes) {
          Node* node = node_ptrs[node_id];
          node->value = old_values[node_id];
        }
        tgt_node->value = old_values[it->first];
        cache_logodds[it->first] = logodds;
      } else {
        // if we change the value of this node then all the other nodes in the
        // pool that depend on this need to be recomputed
        for (uint node_id : markov_blanket[it->first]) {
          cache_logodds[node_id] = NAN;
        }
        cache_logodds[it->first] = -logodds;
      }
    }
    if (infer_config.keep_log_prob) {
      collect_log_prob(_full_log_prob(ordered_supp));
    }
    if (infer_config.keep_warmup or snum >= infer_config.num_warmup) {
      collect_sample();
    }
  }
}

} // namespace graph
} // namespace beanmachine
