// Copyright (c) Facebook, Inc. and its affiliates.
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

void Graph::nmc(uint num_samples, std::mt19937& gen) {
  // convert the smart pointers in nodes to dumb pointers in node_ptrs
  // for faster access
  std::vector<Node*> node_ptrs;
  for (uint node_id = 0; node_id < nodes.size(); node_id++) {
    node_ptrs.push_back(nodes[node_id].get());
  }
  assert(node_ptrs.size() > 0); // keep linter happy
  // eval each node so that we have a starting value and verify that these
  // values are all continuous-valued scalars
  // also compute the pool of variables that we will infer over and
  // compute their descendants -- i.e. all stochastic non-observed nodes
  // that are in the support of the graph
  // pool : nodes that we will infer over -> det_desc, sto_desc
  std::set<uint> supp = compute_support();
  std::map<uint, std::tuple<std::vector<uint>, std::vector<uint>>> pool;
  for (uint node_id : supp) {
    Node* node = node_ptrs[node_id];
    bool node_is_not_observed = observed.find(node_id) == observed.end();
    if (node->is_stochastic() and node_is_not_observed) {
      if (node->value.type != AtomicType::PROBABILITY and
          node->value.type != AtomicType::REAL and
          node->value.type != AtomicType::POS_REAL and
          node->value.type != AtomicType::BOOLEAN) {
        throw std::runtime_error(
            "NMC only supported on bool/probability/real/positive -- failing on node " +
            std::to_string(node_id));
      }
      node->value = proposer::uniform_initializer(gen, node->value.type);
      std::vector<uint> det_nodes;
      std::vector<uint> sto_nodes;
      std::tie(det_nodes, sto_nodes) = compute_descendants(node_id, supp);
      pool[node_id] = std::make_tuple(det_nodes, sto_nodes);
    }
    else if (node_is_not_observed) {
      node->eval(gen); // evaluate the value of non-observed operator nodes
    }
  }
  std::vector<NodeValue> old_values = std::vector<NodeValue>(nodes.size());
  assert(old_values.size() > 0); // keep linter happy
  // sampling outer loop
  for (uint snum = 0; snum < num_samples; snum++) {
    for (auto it = pool.begin(); it != pool.end(); ++it) {
      // for the target sampled node grab its deterministic and stochastic
      // children
      // the following dance of getting into a tuple is needed because this
      // version of C++ doesn't have structured bindings
      std::tuple<const std::vector<uint>&, const std::vector<uint>&> tmp_tuple =
          it->second;
      const std::vector<uint>& det_nodes = std::get<0>(tmp_tuple);
      const std::vector<uint>& sto_nodes = std::get<1>(tmp_tuple);
      assert(it->first == sto_nodes.front());
      // Go through all the children of this node and
      // - propagate gradients
      // - save old values of deterministic nodes
      // - add log_prob of stochastic nodes
      // - add gradient_log_prob of stochastic nodes
      // Note: all gradients are w.r.t. the current node that we are sampling
      Node* tgt_node = node_ptrs[it->first];
      tgt_node->grad1 = 1;
      tgt_node->grad2 = 0;
      for (uint node_id : det_nodes) {
        Node* node = node_ptrs[node_id];
        old_values[node_id] = node->value;
        node->compute_gradients();
      }
      double old_logweight = 0;
      double old_grad1 = 0;
      double old_grad2 = 0;
      for (uint node_id : sto_nodes) {
        const Node* node = node_ptrs[node_id];
        old_logweight += node->log_prob();
        node->gradient_log_prob(old_grad1, old_grad2);
      }
      // now create a proposer object, save the value of tgt_node and propose a
      // new value
      std::unique_ptr<proposer::Proposer> old_prop =
          proposer::nmc_proposer(tgt_node->value, old_grad1, old_grad2);
      graph::NodeValue old_value = tgt_node->value;
      tgt_node->value = old_prop->sample(gen);
      // similar to the above process we will go through all the children and
      // - compute new value of deterministic nodes
      // - propagate gradients
      // - add log_prob of stochastic nodes
      // - add gradient_log_prob of stochastic nodes
      for (uint node_id : det_nodes) {
        Node* node = node_ptrs[node_id];
        node->eval(gen);
        node->compute_gradients();
      }
      double new_logweight = 0;
      double new_grad1 = 0;
      double new_grad2 = 0;
      for (uint node_id : sto_nodes) {
        const Node* node = node_ptrs[node_id];
        new_logweight += node->log_prob();
        node->gradient_log_prob(new_grad1, new_grad2);
      }
      // construct the reverse proposer and use it to compute the
      // log acceptance probability of the move
      std::unique_ptr<proposer::Proposer> new_prop =
          proposer::nmc_proposer(tgt_node->value, new_grad1, new_grad2);
      double logacc = new_logweight - old_logweight +
          new_prop->log_prob(old_value) - old_prop->log_prob(tgt_node->value);
      // The move is accepted if the probability is > 1 or if we sample and get
      // a true Otherwise we reject the move and restore all the deterministic
      // children and the value of the target node. In either case we need to
      // restore the gradients.
      if (logacc > 0 or util::sample_logprob(gen, logacc)) {
        for (uint node_id : det_nodes) {
          Node* node = node_ptrs[node_id];
          node->grad1 = node->grad2 = 0;
        }
      } else {
        for (uint node_id : det_nodes) {
          Node* node = node_ptrs[node_id];
          node->value = old_values[node_id];
          node->grad1 = node->grad2 = 0;
        }
        tgt_node->value = old_value;
      }
      tgt_node->grad1 = tgt_node->grad2 = 0;
    }
    collect_sample();
  }
}

} // namespace graph
} // namespace beanmachine
