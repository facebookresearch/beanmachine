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
  // A map from node id to its deterministic and stochastic operator
  // descendant nodes that are in the support.
  std::map<uint, std::tuple<std::vector<uint>, std::vector<uint>>> pool;

 public:
  NMC(Graph* g, std::mt19937& gen) : g(g), gen(gen) {}

  void infer(uint num_samples) {
    g->pd_begin(ProfilerEvent::NMC_INFER);

    initialize();

    g->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
    assert(old_values.size() > 0); // keep linter happy
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
        } else {
          nmc_step(tgt_node, det_nodes, sto_nodes);
        }
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
  void initialize() {
    g->pd_begin(ProfilerEvent::NMC_INFER_INITIALIZE);
    smart_to_dumb();
    compute_support();
    compute_unobserved_support();
    ensure_continuous();
    compute_initial_values();
    compute_pool();
    compute_ordered_support();
    old_values = std::vector<NodeValue>(g->nodes.size());
    g->pd_finish(ProfilerEvent::NMC_INFER_INITIALIZE);
  }

  void smart_to_dumb() {
    // Convert the smart pointers in nodes to dumb pointers in node_ptrs
    // for faster access.
    for (uint node_id = 0; node_id < g->nodes.size(); node_id++) {
      node_ptrs.push_back(g->nodes[node_id].get());
    }
  }

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

  static bool is_not_supported(Node* node) {
    return node->is_stochastic() and
        node->value.type.variable_type != VariableType::COL_SIMPLEX_MATRIX and
        node->value.type != AtomicType::PROBABILITY and
        node->value.type != AtomicType::REAL and
        node->value.type != AtomicType::POS_REAL and
        node->value.type != AtomicType::BOOLEAN;
  }

  void ensure_continuous() {
    for (uint node_id : unobserved_supp) {
      Node* node = node_ptrs[node_id];
      if (is_not_supported(node)) {
        throw std::runtime_error(
            "NMC only supported on bool/probability/real/positive -- failing on node " +
            std::to_string(node_id));
      }
    }
  }

  void compute_initial_values() {
    for (uint node_id : unobserved_supp) {
      Node* node = node_ptrs[node_id];
      if (node->is_stochastic()) {
        if (node->value.type.variable_type ==
            VariableType::COL_SIMPLEX_MATRIX) {
          auto sto_node = static_cast<oper::StochasticOperator*>(node);
          sto_node->unconstrained_value = sto_node->value;
        } else {
          node->value = proposer::uniform_initializer(gen, node->value.type);
        }
      } else {
        node->eval(gen); // evaluate the value of non-observed operator nodes
      }
    }
  }

  void compute_pool() {
    for (uint node_id : unobserved_supp) {
      if (node_ptrs[node_id]->is_stochastic()) {
        std::vector<uint> det_nodes;
        std::vector<uint> sto_nodes;
        std::tie(det_nodes, sto_nodes) = g->compute_descendants(node_id, supp);
        pool[node_id] = std::make_tuple(det_nodes, sto_nodes);
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

  void save_old_values(const std::vector<uint>& det_nodes) {
    for (uint node_id : det_nodes) {
      Node* node = node_ptrs[node_id];
      old_values[node_id] = node->value;
    }
  }

  void restore_old_values(const std::vector<uint>& det_nodes) {
    for (uint node_id : det_nodes) {
      Node* node = node_ptrs[node_id];
      node->value = old_values[node_id];
    }
  }

  void compute_gradients(const std::vector<uint>& det_nodes) {
    for (uint node_id : det_nodes) {
      Node* node = node_ptrs[node_id];
      node->compute_gradients();
    }
  }

  void clear_gradients(const std::vector<uint>& det_nodes) {
    for (uint node_id : det_nodes) {
      Node* node = node_ptrs[node_id];
      node->grad1 = node->grad2 = 0;
    }
  }

  std::unique_ptr<proposer::Proposer> create_proposer(
      const std::vector<uint>& sto_nodes,
      NodeValue value,
      /* out */ double& logweight) {
    logweight = 0;
    double grad1 = 0;
    double grad2 = 0;
    for (uint node_id : sto_nodes) {
      const Node* node = node_ptrs[node_id];
      logweight += node->log_prob();
      node->gradient_log_prob(/* in-out */ grad1, /* in-out */ grad2);
    }
    std::unique_ptr<proposer::Proposer> prop =
        proposer::nmc_proposer(value, grad1, grad2);
    return prop;
  }

  std::unique_ptr<proposer::Proposer> create_proposer_dirichlet(
      const std::vector<uint>& sto_nodes,
      Node* tgt_node,
      double param_a,
      NodeValue value,
      /* out */ double& logweight) {
    logweight = 0;
    double grad1 = 0;
    double grad2 = 0;
    for (uint node_id : sto_nodes) {
      const Node* node = node_ptrs[node_id];
      if (node == tgt_node) {
        // X_k ~ Gamma(param_a, 1)
        logweight += (param_a - 1.0) * std::log(value._double) - value._double -
            lgamma(param_a);
        grad1 += (param_a - 1.0) / value._double - 1.0;
        grad2 += (1.0 - param_a) / (value._double * value._double);
      } else {
        logweight += node->log_prob();
        node->gradient_log_prob(/* in-out */ grad1, /* in-out */ grad2);
      }
    }
    std::unique_ptr<proposer::Proposer> prop =
        proposer::nmc_proposer(value, grad1, grad2);
    return prop;
  }

  void nmc_step(
      Node* tgt_node,
      const std::vector<uint>& det_nodes,
      const std::vector<uint>& sto_nodes) {
    tgt_node->grad1 = 1;
    tgt_node->grad2 = 0;
    NodeValue old_value = tgt_node->value;
    save_old_values(det_nodes);
    compute_gradients(det_nodes);

    // Propose a new value.
    double old_logweight;
    auto old_prop =
        create_proposer(sto_nodes, old_value, /* out */ old_logweight);
    NodeValue new_value = old_prop->sample(gen);

    // similar to the above process we will go through all the children and
    // - compute new value of deterministic nodes
    // - propagate gradients
    // - add log_prob of stochastic nodes
    // - add gradient_log_prob of stochastic nodes
    tgt_node->value = new_value;
    for (uint node_id : det_nodes) {
      Node* node = node_ptrs[node_id];
      node->eval(gen);
      node->compute_gradients();
    }
    // construct the reverse proposer and use it to compute the
    // log acceptance probability of the move
    double new_logweight;
    auto new_prop =
        create_proposer(sto_nodes, new_value, /* out */ new_logweight);
    double logacc = new_logweight - old_logweight +
        new_prop->log_prob(old_value) - old_prop->log_prob(new_value);
    // The move is accepted if the probability is > 1 or if we sample and
    // get a true Otherwise we reject the move and restore all the
    // deterministic children and the value of the target node. In either
    // case we need to restore the gradients.
    bool accepted = logacc > 0 or util::sample_logprob(gen, logacc);
    if (!accepted) {
      restore_old_values(det_nodes);
      tgt_node->value = old_value;
    }
    clear_gradients(det_nodes);
    tgt_node->grad1 = tgt_node->grad2 = 0;
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
      NodeValue old_value(AtomicType::POS_REAL, old_X_k);
      save_old_values(det_nodes);
      compute_gradients(det_nodes);
      double old_logweight;
      auto old_prop = create_proposer_dirichlet(
          sto_nodes, tgt_node, param_a, old_value, /* out */ old_logweight);
      NodeValue new_value = old_prop->sample(gen);
      *(src_node->unconstrained_value._matrix.data() + k) = new_value._double;
      sum = src_node->unconstrained_value._matrix.sum();
      src_node->value._matrix =
          src_node->unconstrained_value._matrix.array() / sum;

      // Propagate values and gradients at new value of X_k
      src_node->Grad1 =
          -src_node->unconstrained_value._matrix.array() / (sum * sum);
      *(src_node->Grad1.data() + k) += 1 / sum;
      src_node->Grad2 = src_node->Grad1 * (-2.0) / sum;
      for (uint node_id : det_nodes) {
        Node* node = node_ptrs[node_id];
        node->eval(gen);
        node->compute_gradients();
      }
      double new_logweight;
      auto new_prop = create_proposer_dirichlet(
          sto_nodes, tgt_node, param_a, new_value, /* out */ new_logweight);
      double logacc = new_logweight - old_logweight +
          new_prop->log_prob(old_value) - old_prop->log_prob(new_value);
      // Accept or reject, reset (values and) gradients
      bool accepted = logacc > 0 or util::sample_logprob(gen, logacc);
      if (!accepted) {
        restore_old_values(det_nodes);
        *(src_node->unconstrained_value._matrix.data() + k) = old_X_k;
        sum = src_node->unconstrained_value._matrix.sum();
        src_node->value._matrix =
            src_node->unconstrained_value._matrix.array() / sum;
      }
      clear_gradients(det_nodes);
      tgt_node->grad1 = tgt_node->grad2 = 0;
    }
  }
};

void Graph::nmc(uint num_samples, std::mt19937& gen) {
  Graph::NMC(this, gen).infer(num_samples);
}

} // namespace graph
} // namespace beanmachine
