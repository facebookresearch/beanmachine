// Copyright (c) Facebook, Inc. and its affiliates.
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

class Graph::NMC {
 private:
  Graph* g;
  std::mt19937& gen;
  // Map from node id to Node*.
  std::vector<Node*> node_ptrs;
  std::vector<NodeValue> old_values;
  // IDs of all nodes in the graph that are directly or
  // indirectly observed or queried.
  std::set<uint> supp;
  std::vector<Node*> ordered_supp;
  // Nodes in supp that are not directly observed.
  std::set<uint> unobserved_supp;

 public:
  NMC(Graph* g, std::mt19937& gen) : g(g), gen(gen) {}

  void infer(uint num_samples) {
    g->pd_begin(ProfilerEvent::NMC_INFER);
    // convert the smart pointers in nodes to dumb pointers in node_ptrs
    // for faster access
    g->pd_begin(ProfilerEvent::NMC_INFER_INITIALIZE);

    for (uint node_id = 0; node_id < g->nodes.size(); node_id++) {
      node_ptrs.push_back(g->nodes[node_id].get());
    }

    compute_support();
    compute_unobserved_support();

    // eval each node so that we have a starting value and verify that these
    // values are all continuous-valued scalars
    // also compute the pool of variables that we will infer over and
    // compute their descendants -- i.e. all stochastic non-observed nodes
    // that are in the support of the graph
    // pool : nodes that we will infer over -> det_desc, sto_desc

    std::map<uint, std::tuple<std::vector<uint>, std::vector<uint>>> pool;
    std::vector<Node*> ordered_supp;
    for (uint node_id : unobserved_supp) {
      // @lint-ignore CLANGTIDY
      Node* node = node_ptrs[node_id];
      if (node->is_stochastic()) {
        if (node->value.type.variable_type ==
            VariableType::COL_SIMPLEX_MATRIX) {
          auto sto_node = static_cast<oper::StochasticOperator*>(node);
          sto_node->unconstrained_value = sto_node->value;
        } else {
          if (node->value.type != AtomicType::PROBABILITY and
              node->value.type != AtomicType::REAL and
              node->value.type != AtomicType::POS_REAL and
              node->value.type != AtomicType::BOOLEAN) {
            throw std::runtime_error(
                "NMC only supported on bool/probability/real/positive -- failing on node " +
                std::to_string(node_id));
          }
          node->value = proposer::uniform_initializer(gen, node->value.type);
        }
        std::vector<uint> det_nodes;
        std::vector<uint> sto_nodes;
        std::tie(det_nodes, sto_nodes) = g->compute_descendants(node_id, supp);
        pool[node_id] = std::make_tuple(det_nodes, sto_nodes);
      } else {
        node->eval(gen); // evaluate the value of non-observed operator nodes
      }
    }

    compute_ordered_support();

    g->pd_finish(ProfilerEvent::NMC_INFER_INITIALIZE);
    g->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
    old_values = std::vector<NodeValue>(g->nodes.size());
    // sampling outer loop
    for (uint snum = 0; snum < num_samples; snum++) {
      for (auto it = pool.begin(); it != pool.end(); ++it) {
        // for the target sampled node grab its deterministic and stochastic
        // children
        // the following dance of getting into a tuple is needed because this
        // version of C++ doesn't have structured bindings
        std::tuple<const std::vector<uint>&, const std::vector<uint>&>
            tmp_tuple = it->second;
        const std::vector<uint>& det_nodes = std::get<0>(tmp_tuple);
        const std::vector<uint>& sto_nodes = std::get<1>(tmp_tuple);
        assert(it->first == sto_nodes.front());
        // Go through all the children of this node and
        // - propagate gradients
        // - save old values of deterministic nodes
        // - add log_prob of stochastic nodes
        // - add gradient_log_prob of stochastic nodes
        // Note: all gradients are w.r.t. the current node that we are sampling
        // @lint-ignore CLANGTIDY
        Node* tgt_node = node_ptrs[it->first];
        if (tgt_node->value.type.variable_type ==
            VariableType::COL_SIMPLEX_MATRIX) {
          nmc_step_for_dirichlet(tgt_node, det_nodes, sto_nodes);
          continue;
        }
        tgt_node->grad1 = 1;
        tgt_node->grad2 = 0;
        for (uint node_id : det_nodes) {
          // @lint-ignore CLANGTIDY
          Node* node = node_ptrs[node_id];
          // @lint-ignore CLANGTIDY
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
        // now create a proposer object, save the value of tgt_node and propose
        // a new value
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
        // The move is accepted if the probability is > 1 or if we sample and
        // get a true Otherwise we reject the move and restore all the
        // deterministic children and the value of the target node. In either
        // case we need to restore the gradients.
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
      if (g->infer_config.keep_log_prob) {
        g->collect_log_prob(g->_full_log_prob(ordered_supp));
      }
      g->collect_sample();
    }
    g->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
    g->pd_finish(ProfilerEvent::NMC_INFER);
  }

 private:
  void compute_support() {
    supp = g->compute_support();
  }

  void compute_unobserved_support() {
    for (uint node_id : supp) {
      bool node_is_not_observed =
          g->observed.find(node_id) == g->observed.end();
      if (node_is_not_observed) {
        unobserved_supp.insert(node_id);
      }
    }
  }

  void compute_ordered_support() {
    if (g->infer_config.keep_log_prob) {
      for (uint node_id : supp) {
        ordered_supp.push_back(node_ptrs[node_id]);
      }
    }
  }
  /*
  We treat the K-dimensional Dirichlet sample as K independent Gamma samples
  divided by their sum. i.e. Let X_k ~ Gamma(alpha_k, 1), for k = 1, ..., K,
  Y_k = X_k / sum(X), then (Y_1, ..., Y_K) ~ Dirichlet(alphas). We store Y in
  the attribute value, and X in unconstrainted_value.
  */
  void nmc_step_for_dirichlet(
      Node* tgt_node,
      const std::vector<uint>& det_nodes,
      const std::vector<uint>& sto_nodes) {
    uint K = tgt_node->value._matrix.size();
    auto src_node = static_cast<oper::StochasticOperator*>(tgt_node);
    // @lint-ignore CLANGTIDY
    auto param_node = src_node->in_nodes[0]->in_nodes[0];
    double param_a, old_X_k, sum;
    for (uint k = 0; k < K; k++) {
      // Prepare gradients
      // Grad1 = (dY_1/dX_k, dY_2/dX_k, ..., dY_K/X_k)
      // where dY_k/dX_k = (sum(X) - X_k)/sum(X)^2
      //       dY_j/dX_k = - X_j/sum(X)^2, for j != k
      // Grad2 = (d^2Y_1/dX^2_k, ..., d^2Y_K/X^2_k)
      // where d2Y_k/dX2_k = -2 * (sum(X) - X_k)/sum(X)^3
      //       d2Y_j/dX2_k = -2 * X_j/sum(X)^3
      param_a = param_node->value._matrix.coeff(k);
      old_X_k = src_node->unconstrained_value._matrix.coeff(k);
      sum = src_node->unconstrained_value._matrix.sum();
      src_node->Grad1 =
          -src_node->unconstrained_value._matrix.array() / (sum * sum);
      *(src_node->Grad1.data() + k) += 1 / sum;
      src_node->Grad2 = src_node->Grad1 * (-2.0) / sum;
      src_node->grad1 = 1;
      src_node->grad2 = 0;
      // Propagate gradients
      for (uint node_id : det_nodes) {
        // @lint-ignore CLANGTIDY
        Node* node = node_ptrs[node_id];
        // @lint-ignore CLANGTIDY
        old_values[node_id] = node->value;
        node->compute_gradients();
      }
      double old_logweight = 0;
      double old_grad1 = 0;
      double old_grad2 = 0;
      for (uint node_id : sto_nodes) {
        const Node* node = node_ptrs[node_id];
        if (node == tgt_node) {
          // X_k ~ Gamma(param_a, 1)
          old_logweight +=
              (param_a - 1.0) * std::log(old_X_k) - old_X_k - lgamma(param_a);
          old_grad1 += (param_a - 1.0) / old_X_k - 1.0;
          old_grad2 += (1.0 - param_a) / (old_X_k * old_X_k);
        } else {
          old_logweight += node->log_prob();
          node->gradient_log_prob(old_grad1, old_grad2);
        }
      }
      // Create forward(old) proposer, propose new value of X_k
      NodeValue old_value(AtomicType::POS_REAL, old_X_k);
      std::unique_ptr<proposer::Proposer> old_prop =
          proposer::nmc_proposer(old_value, old_grad1, old_grad2);
      NodeValue new_value = old_prop->sample(gen);
      *(src_node->unconstrained_value._matrix.data() + k) = new_value._double;
      sum = src_node->unconstrained_value._matrix.sum();
      src_node->value._matrix =
          src_node->unconstrained_value._matrix.array() / sum;

      // Prapagate values and gradients at new value of X_k
      src_node->Grad1 =
          -src_node->unconstrained_value._matrix.array() / (sum * sum);
      *(src_node->Grad1.data() + k) += 1 / sum;
      src_node->Grad2 = src_node->Grad1 * (-2.0) / sum;
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
        if (node == tgt_node) {
          // X_k ~ Gamma(param_a, 1)
          new_logweight += (param_a - 1.0) * std::log(new_value._double) -
              new_value._double - lgamma(param_a);
          new_grad1 += (param_a - 1.0) / new_value._double - 1.0;
          new_grad2 +=
              (1.0 - param_a) / (new_value._double * new_value._double);
        } else {
          new_logweight += node->log_prob();
          node->gradient_log_prob(new_grad1, new_grad2);
        }
      }
      // Create the reverse(new) proposer
      std::unique_ptr<proposer::Proposer> new_prop =
          proposer::nmc_proposer(new_value, new_grad1, new_grad2);
      double logacc = new_logweight - old_logweight +
          new_prop->log_prob(old_value) - old_prop->log_prob(new_value);
      // Accept or reject, reset (values and) gradients
      if (logacc > 0 or util::sample_logprob(gen, logacc)) {
        // accepted:
        for (uint node_id : det_nodes) {
          Node* node = node_ptrs[node_id];
          node->grad1 = node->grad2 = 0;
        }
      } else {
        // rejected:
        for (uint node_id : det_nodes) {
          // @lint-ignore CLANGTIDY
          Node* node = node_ptrs[node_id];
          // @lint-ignore CLANGTIDY
          node->value = old_values[node_id];
          node->grad1 = node->grad2 = 0;
        }
        *(src_node->unconstrained_value._matrix.data() + k) = old_X_k;
        sum = src_node->unconstrained_value._matrix.sum();
        src_node->value._matrix =
            src_node->unconstrained_value._matrix.array() / sum;
      }
      tgt_node->grad1 = tgt_node->grad2 = 0;
    }
  }
};

void Graph::nmc(uint num_samples, std::mt19937& gen) {
  Graph::NMC(this, gen).infer(num_samples);
}

} // namespace graph
} // namespace beanmachine
