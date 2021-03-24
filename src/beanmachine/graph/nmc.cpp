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

  // A graph maintains of a vector of nodes; the index into that vector is
  // the id of the node. We often need to translate from node ids into node
  // pointers in this algorithm; to do so quickly we obtain the address of
  // every node in the graph up front and then look it up when we need it.
  std::vector<Node*> node_ptrs;

  // Every node in the graph has a value; when we propose a new graph state,
  // we update the values. If we then reject the proposed new state, we need
  // to restore the values. This vector stores the original values of the
  // nodes that we change during the proposal step.
  std::vector<NodeValue> old_values;

  // The support is the set of all nodes in the graph that are queried or
  // observed, directly or indirectly. We need both the support as nodes
  // and as pointers in this algorithm.
  std::set<uint> supp_ids;
  std::vector<Node*> supp;

  // Nodes in supp that are not directly observed.  Note that
  // the order of nodes in this vector matters! We must enumerate
  // them in order from lowest node identifier to highest.
  std::vector<Node*> unobserved_supp;

  // Nodes in unobserved_supp that are stochastic; similarly, order matters.
  std::vector<Node*> unobserved_sto_supp;

  // These vectors are the same size as unobserved_sto_support.
  // The elements are vectors of nodes; those nodes are in
  // the support and are the stochastic or deterministic
  // descendents of the corresponding unobserved stochastic node.
  std::vector<std::vector<Node*>> sto_descendants;
  std::vector<std::vector<Node*>> det_descendants;

 public:
  NMC(Graph* g, std::mt19937& gen) : g(g), gen(gen) {}

  void infer(uint num_samples) {
    g->pd_begin(ProfilerEvent::NMC_INFER);
    initialize();
    collect_samples(num_samples);
    g->pd_finish(ProfilerEvent::NMC_INFER);
  }

 private:
  // The initialization phase precomputes the vectors we are going to
  // need during inference, and verifies that the NMC algorithm can
  // compute gradients of every node we need to.
  void initialize() {
    g->pd_begin(ProfilerEvent::NMC_INFER_INITIALIZE);
    smart_to_dumb();
    compute_support();
    ensure_continuous();
    compute_initial_values();
    compute_descendants();
    old_values = std::vector<NodeValue>(g->nodes.size());
    g->pd_finish(ProfilerEvent::NMC_INFER_INITIALIZE);
  }

  void smart_to_dumb() {
    for (uint node_id = 0; node_id < g->nodes.size(); node_id++) {
      node_ptrs.push_back(g->nodes[node_id].get());
    }
  }

  void compute_support() {
    supp_ids = g->compute_support();
    for (uint node_id : supp_ids) {
      supp.push_back(node_ptrs[node_id]);
    }
    for (Node* node : supp) {
      bool node_is_not_observed =
          g->observed.find(node->index) == g->observed.end();
      if (node_is_not_observed) {
        unobserved_supp.push_back(node);
        if (node->is_stochastic()) {
          unobserved_sto_supp.push_back(node);
        }
      }
    }
  }

  static bool is_not_supported(Node* node) {
    return node->value.type.variable_type !=
        VariableType::COL_SIMPLEX_MATRIX and
        node->value.type != AtomicType::PROBABILITY and
        node->value.type != AtomicType::REAL and
        node->value.type != AtomicType::POS_REAL and
        node->value.type != AtomicType::BOOLEAN;
  }

  void ensure_continuous() {
    for (Node* node : unobserved_sto_supp) {
      if (is_not_supported(node)) {
        throw std::runtime_error(
            "NMC only supported on bool/probability/real/positive -- failing on node " +
            std::to_string(node->index));
      }
    }
  }

  // We can now compute the initial state of the graph. Observed nodes
  // will have values given by the observation, so we can ignore those.
  // Unobserved stochastic nodes are assigned a value by the uniform
  // initializer. Deterministic nodes are computed from their inputs.
  void compute_initial_values() {
    for (Node* node : unobserved_supp) {
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

  // For every unobserved stochastic node in the graph, we will need to
  // repeatedly know the set of deterministic and stochastic descendant
  // nodes. Because this can be expensive, we compute those sets once
  // and cache them.
  void compute_descendants() {
    for (Node* node : unobserved_sto_supp) {
      std::vector<uint> det_node_ids;
      std::vector<uint> sto_node_ids;
      std::vector<Node*> det_nodes;
      std::vector<Node*> sto_nodes;
      std::tie(det_node_ids, sto_node_ids) =
          g->compute_descendants(node->index, supp_ids);
      for (uint id : det_node_ids) {
        det_nodes.push_back(node_ptrs[id]);
      }
      for (uint id : sto_node_ids) {
        sto_nodes.push_back(node_ptrs[id]);
      }
      det_descendants.push_back(det_nodes);
      sto_descendants.push_back(sto_nodes);
    }
  }

  void generate_sample() {
    for (uint i = 0; i < unobserved_sto_supp.size(); ++i) {
      Node* tgt_node = unobserved_sto_supp[i];
      const std::vector<Node*>& det_nodes = det_descendants[i];
      const std::vector<Node*>& sto_nodes = sto_descendants[i];
      assert(tgt_node == sto_nodes.front());
      if (tgt_node->value.type.variable_type ==
          VariableType::COL_SIMPLEX_MATRIX) {
        nmc_step_for_dirichlet(tgt_node, det_nodes, sto_nodes);
      } else {
        nmc_step(tgt_node, det_nodes, sto_nodes);
      }
    }
  }

  void collect_samples(uint num_samples) {
    g->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
    for (uint snum = 0; snum < num_samples; snum++) {
      generate_sample();
      collect_sample();
    }
    g->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
  }

  void collect_sample() {
    if (g->infer_config.keep_log_prob) {
      g->collect_log_prob(g->_full_log_prob(supp));
    }
    g->collect_sample();
  }

  void save_old_values(const std::vector<Node*>& det_nodes) {
    for (Node* node : det_nodes) {
      old_values[node->index] = node->value;
    }
  }

  void restore_old_values(const std::vector<Node*>& det_nodes) {
    for (Node* node : det_nodes) {
      node->value = old_values[node->index];
    }
  }

  void compute_gradients(const std::vector<Node*>& det_nodes) {
    for (Node* node : det_nodes) {
      node->compute_gradients();
    }
  }

  void eval(const std::vector<Node*>& det_nodes) {
    for (Node* node : det_nodes) {
      node->eval(gen);
    }
  }

  void clear_gradients(const std::vector<Node*>& det_nodes) {
    for (Node* node : det_nodes) {
      node->grad1 = 0;
      node->grad2 = 0;
    }
  }

  // This method performs two tasks:
  // * Compute the "score" for a given collection of stochastic
  //   nodes; that is, how likely is this combination of samples?
  // * Create a proposer that can randomly choose a new value for a node
  //   based on the current value and the gradients of the stochastic
  //   nodes.
  std::unique_ptr<proposer::Proposer> create_proposer(
      const std::vector<Node*>& sto_nodes,
      NodeValue value,
      /* out */ double& logweight) {
    logweight = 0;
    double grad1 = 0;
    double grad2 = 0;
    for (Node* node : sto_nodes) {
      logweight += node->log_prob();
      node->gradient_log_prob(/* in-out */ grad1, /* in-out */ grad2);
    }
    std::unique_ptr<proposer::Proposer> prop =
        proposer::nmc_proposer(value, grad1, grad2);
    return prop;
  }

  std::unique_ptr<proposer::Proposer> create_proposer_dirichlet(
      const std::vector<Node*>& sto_nodes,
      Node* tgt_node,
      double param_a,
      NodeValue value,
      /* out */ double& logweight) {
    logweight = 0;
    double grad1 = 0;
    double grad2 = 0;
    for (Node* node : sto_nodes) {
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
      const std::vector<Node*>& det_nodes,
      const std::vector<Node*>& sto_nodes) {
    g->pd_begin(ProfilerEvent::NMC_STEP);
    // We are given an unobserved stochastic "target" node and we wish
    // to compute a new value for it. The basic idea of the algorithm is:
    //
    // * Save the current state of the graph.
    // * Compute the "score" of the current state: how likely is this state?
    // * Propose a new value for the target node.
    // * Changing the value of the target node changes the values and gradients
    //   of its descendents; make those updates.
    // * Evaluate the "score" of the proposed new state.
    // * Accept or reject the proposed new value based on the relative scores.
    // * If we rejected it, restore the saved state.

    NodeValue old_value = tgt_node->value;
    save_old_values(det_nodes);

    tgt_node->grad1 = 1;
    tgt_node->grad2 = 0;
    // Deterministic node values are already evaluated but gradients
    // are not.
    compute_gradients(det_nodes);
    double old_logweight;
    auto old_prop =
        create_proposer(sto_nodes, old_value, /* out */ old_logweight);

    NodeValue new_value = old_prop->sample(gen);

    tgt_node->value = new_value;
    eval(det_nodes);
    compute_gradients(det_nodes);
    double new_logweight;
    auto new_prop =
        create_proposer(sto_nodes, new_value, /* out */ new_logweight);

    double logacc = new_logweight - old_logweight +
        new_prop->log_prob(old_value) - old_prop->log_prob(new_value);
    bool accepted = logacc > 0 or util::sample_logprob(gen, logacc);
    if (!accepted) {
      restore_old_values(det_nodes);
      tgt_node->value = old_value;
    }

    clear_gradients(det_nodes);
    tgt_node->grad1 = 0;
    tgt_node->grad2 = 0;
    g->pd_finish(ProfilerEvent::NMC_STEP);
  }

  /*
  We treat the K-dimensional Dirichlet sample as K independent Gamma samples
  divided by their sum. i.e. Let X_k ~ Gamma(alpha_k, 1), for k = 1, ..., K,
  Y_k = X_k / sum(X), then (Y_1, ..., Y_K) ~ Dirichlet(alphas). We store Y in
  the attribute value, and X in unconstrainted_value.
  */
  void nmc_step_for_dirichlet(
      Node* tgt_node,
      const std::vector<Node*>& det_nodes,
      const std::vector<Node*>& sto_nodes) {
    g->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);
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
      eval(det_nodes);
      compute_gradients(det_nodes);

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
      tgt_node->grad1 = 0;
      tgt_node->grad2 = 0;
    } // k
    g->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
  }
};

void Graph::nmc(uint num_samples, std::mt19937& gen) {
  Graph::NMC(this, gen).infer(num_samples);
}

} // namespace graph
} // namespace beanmachine
