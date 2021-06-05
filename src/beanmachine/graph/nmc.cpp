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
#include "beanmachine/graph/proposer/default_initializer.h"
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

  // Nodes in supp that are not directly observed. Note that
  // the order of nodes in this vector matters! We must enumerate
  // them in order from lowest node identifier to highest.
  std::vector<Node*> unobserved_supp;

  // Nodes in unobserved_supp that are stochastic; similarly, order matters.
  std::vector<Node*> unobserved_sto_supp;

  // These vectors are the same size as unobserved_sto_support.
  // The i-th elements are vectors of nodes which are
  // respectively the vector of
  // the immediate stochastic descendants of node with index i in the support,
  // and the vector of the intervening deterministic nodes
  // between the i-th node and its immediate stochastic descendants.
  // In other words, these are the cached results of
  // invoking graph::get_nodes_up_to_immediate_descendants
  // for each node.
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
    collect_node_ptrs();
    compute_support();
    ensure_continuous();
    compute_initial_values();
    compute_affected_nodes();
    old_values = std::vector<NodeValue>(g->nodes.size());
    g->pd_finish(ProfilerEvent::NMC_INFER_INITIALIZE);
  }

  void collect_node_ptrs() {
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

  static bool is_not_supported(Node* node) { // specific to NMC
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
  // Note that we only need a single pass because parent nodes always have
  // indices less than those of their children, and unobserved_supp
  // respects index order.
  void compute_initial_values() {
    for (Node* unobs_node : unobserved_supp) {
      if (unobs_node->is_stochastic()) {
        proposer::default_initializer(gen, unobs_node);
      } else { // non-stochastic operator node, so just evaluate
        unobs_node->eval(gen);
      }
    }
  }

  // For every unobserved stochastic node in the graph, we will need to
  // repeatedly know the set of immediate stochastic descendants
  // and intervening deterministic nodes.
  // Because this can be expensive, we compute those sets once and cache them.
  void compute_affected_nodes() {
    for (Node* node : unobserved_sto_supp) {
      std::vector<uint> det_node_ids;
      std::vector<uint> sto_node_ids;
      std::vector<Node*> det_nodes;
      std::vector<Node*> sto_nodes;
      std::tie(det_node_ids, sto_node_ids) =
          g->compute_affected_nodes(node->index, supp_ids);
      for (uint id : det_node_ids) {
        det_nodes.push_back(node_ptrs[id]);
      }
      for (uint id : sto_node_ids) {
        sto_nodes.push_back(node_ptrs[id]);
      }
      det_descendants.push_back(det_nodes);
      sto_descendants.push_back(sto_nodes);
      if (g->_collect_performance_data) {
        g->profiler_data.det_supp_count[node->index] = det_nodes.size();
      }
    }
  }

  void generate_sample() {
    for (uint i = 0; i < unobserved_sto_supp.size(); ++i) {
      Node* tgt_node = unobserved_sto_supp[i];
      // TODO sto_nodes does not seem to be defined anywhere.
      // It looks like assert is disabled, but this will not compile
      // once it is enabled.
      assert(tgt_node == sto_nodes.front());
      if (tgt_node->value.type.variable_type ==
          VariableType::COL_SIMPLEX_MATRIX) { // TODO make more generic
        if (tgt_node->value.type.rows == 2) {
          nmc_step_for_dirichlet_beta(
              tgt_node, det_descendants[i], sto_descendants[i]);
        } else {
          nmc_step_for_dirichlet_gamma(
              tgt_node, det_descendants[i], sto_descendants[i]);
        }
      } else {
        mh_step(tgt_node, det_descendants[i], sto_descendants[i]);
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
    g->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLE);
    if (g->infer_config.keep_log_prob) {
      g->collect_log_prob(g->_full_log_prob(supp));
    }
    g->collect_sample();
    g->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLE);
  }

  void save_old_values(const std::vector<Node*>& det_nodes) {
    g->pd_begin(ProfilerEvent::NMC_SAVE_OLD);
    for (Node* node : det_nodes) {
      old_values[node->index] = node->value;
    }
    g->pd_finish(ProfilerEvent::NMC_SAVE_OLD);
  }

  void restore_old_values(const std::vector<Node*>& det_nodes) {
    g->pd_begin(ProfilerEvent::NMC_RESTORE_OLD);
    for (Node* node : det_nodes) {
      node->value = old_values[node->index];
    }
    g->pd_finish(ProfilerEvent::NMC_RESTORE_OLD);
  }

  void compute_gradients(const std::vector<Node*>& det_nodes) {
    g->pd_begin(ProfilerEvent::NMC_COMPUTE_GRADS);
    for (Node* node : det_nodes) {
      node->compute_gradients();
    }
    g->pd_finish(ProfilerEvent::NMC_COMPUTE_GRADS);
  }

  void eval(const std::vector<Node*>& det_nodes) {
    g->pd_begin(ProfilerEvent::NMC_EVAL);
    for (Node* node : det_nodes) {
      node->eval(gen);
    }
    g->pd_finish(ProfilerEvent::NMC_EVAL);
  }

  void clear_gradients(const std::vector<Node*>& det_nodes) {
    g->pd_begin(ProfilerEvent::NMC_CLEAR_GRADS);
    for (Node* node : det_nodes) {
      node->grad1 = 0;
      node->grad2 = 0;
    }
    g->pd_finish(ProfilerEvent::NMC_CLEAR_GRADS);
  }

  // Computes the log probability with respect to a given
  // set of stochastic nodes.
  double compute_log_prob_of(const std::vector<Node*>& sto_nodes) {
    double log_prob = 0;
    for (Node* node : sto_nodes) {
      log_prob += node->log_prob();
    }
    return log_prob;
  }

  // Returns the NMC proposal distribution conditioned on the
  // target node's current value.
  // NOTE: assumes that det_descendants's values are already
  // evaluated according to the target node's value.
  std::unique_ptr<proposer::Proposer> get_proposal_distribution(
      Node* tgt_node,
      NodeValue value,
      const std::vector<Node*>& det_descendants,
      const std::vector<Node*>& sto_descendants) {
    g->pd_begin(ProfilerEvent::NMC_CREATE_PROP);

    tgt_node->grad1 = 1;
    tgt_node->grad2 = 0;
    compute_gradients(det_descendants);

    double grad1 = 0;
    double grad2 = 0;
    for (Node* node : sto_descendants) {
      node->gradient_log_prob(/* in-out */ grad1, /* in-out */ grad2);
    }

    // TODO: generalize so it works with any proposer, not just nmc_proposer:
    std::unique_ptr<proposer::Proposer> prop =
        proposer::nmc_proposer(value, grad1, grad2);
    g->pd_finish(ProfilerEvent::NMC_CREATE_PROP);
    return prop;
  }

  std::unique_ptr<proposer::Proposer> create_proposer_dirichlet_gamma(
      const std::vector<Node*>& sto_nodes,
      Node* tgt_node,
      double param_a,
      NodeValue value,
      /* out */ double& logweight) {
    g->pd_begin(ProfilerEvent::NMC_CREATE_PROP_DIR);
    logweight = 0;
    double grad1 = 0;
    double grad2 = 0;
    for (Node* node : sto_nodes) {
      if (node == tgt_node) {
        // TODO: unify this computation of logweight
        // and grad1, grad2 with those present in methods
        // log_prob and gradient_log_prob

        // X_k ~ Gamma(param_a, 1)
        // PDF of Gamma(a, 1) is x^(a - 1)exp(-x)/gamma(a)
        // so log pdf(x) = log(x^(a - 1)) + (-x) - log(gamma(a))
        // = (a - 1)*log(x) - x - log(gamma(a))
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
    g->pd_finish(ProfilerEvent::NMC_CREATE_PROP_DIR);
    return prop;
  }

  std::unique_ptr<proposer::Proposer> create_proposer_dirichlet_beta(
      const std::vector<Node*>& sto_nodes,
      Node* tgt_node,
      double param_a,
      double param_b,
      NodeValue value,
      /* out */ double& logweight) {
    logweight = 0;
    double grad1 = 0;
    double grad2 = 0;
    for (Node* node : sto_nodes) {
      if (node == tgt_node) {
        double x = value._double;
        // X_k ~ Beta(param_a, param_b)
        logweight += (param_a - 1) * log(x) + (param_b - 1) * log(1 - x) +
            lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);

        grad1 += (param_a - 1) / x - (param_b - 1) / (1 - x);
        grad2 += -(param_a - 1) / (x * x) - (param_b - 1) / ((1 - x) * (1 - x));
      } else {
        logweight += node->log_prob();
        node->gradient_log_prob(/* in-out */ grad1, /* in-out */ grad2);
      }
    }

    std::unique_ptr<proposer::Proposer> prop =
        proposer::nmc_proposer(value, grad1, grad2);
    return prop;
  }

  NodeValue sample(const std::unique_ptr<proposer::Proposer>& prop) {
    g->pd_begin(ProfilerEvent::NMC_SAMPLE);
    NodeValue v = prop->sample(gen);
    g->pd_finish(ProfilerEvent::NMC_SAMPLE);
    return v;
  }

  void mh_step(
      Node* tgt_node,
      const std::vector<Node*>& det_descendants,
      const std::vector<Node*>& sto_descendants) {
    g->pd_begin(ProfilerEvent::NMC_STEP);
    // Implements a Metropolis-Hastings step using the NMC proposer.
    //
    // We are given an unobserved stochastic "target" node and we wish
    // to compute a new value for it. The basic idea of the algorithm is:
    //
    // * Save the current state of the graph. Only deterministic nodes need be
    //   saved because stochastic nodes values may are in principle
    //   compatible with any values of other nodes.
    // * Compute the probability of the current state.
    //   Note that we only need the probability of the immediate stochastic
    //   descendants of the target node, since those are the only ones
    //   whose probability changes when its value is changed
    //   (the probabilities of other nodes becomes irrelevant since
    //   it gets canceled out in the acceptable probability calculation,
    //   as explained below).
    // * Obtains the proposal distribution (old_prop) conditioned on
    //   target node's initial ('old') value.
    // * Propose a new value for the target node.
    // * Evaluate the probability of the proposed new state.
    //   Again, only immediate stochastic nodes need be considered.
    // * Obtains the proposal distribution (new_prop) conditioned on
    //   target node's new value.
    // * Accept or reject the proposed new value based on the
    //   Metropolis-Hastings acceptance probability:
    //          P(new state) * P_new_prop(old state | new state)
    //   min(1, ------------------------------------------------ )
    //          P(old state) * P_old_prop(new state | old state)
    //   but note how the probabilities for the states only need to include
    //   the immediate stochastic descendants since the distributions
    //   are factorized and the remaining stochastic nodes have
    //   their probabilities unchanged and cancel out.
    // * If we rejected it, restore the saved state.

    NodeValue old_value = tgt_node->value;
    save_old_values(det_descendants);

    double old_sto_descendants_log_prob = compute_log_prob_of(sto_descendants);
    auto proposal_distribution_given_old_value = get_proposal_distribution(
        tgt_node, old_value, det_descendants, sto_descendants);

    NodeValue new_value = sample(proposal_distribution_given_old_value);

    tgt_node->value = new_value;
    eval(det_descendants);

    double new_sto_descendants_log_prob = compute_log_prob_of(sto_descendants);
    auto proposal_distribution_given_new_value = get_proposal_distribution(
        tgt_node, new_value, det_descendants, sto_descendants);

    double logacc = new_sto_descendants_log_prob -
        old_sto_descendants_log_prob +
        proposal_distribution_given_new_value->log_prob(old_value) -
        proposal_distribution_given_old_value->log_prob(new_value);

    bool accepted = logacc > 0 or util::sample_logprob(gen, logacc);
    if (!accepted) {
      restore_old_values(det_descendants);
      tgt_node->value = old_value;
    }

    // TODO clarify why it is necessary to clear the gradients
    // since we seem to be computing them from scratch when we need them.
    clear_gradients(det_descendants);
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
  void nmc_step_for_dirichlet_gamma(
      Node* tgt_node,
      const std::vector<Node*>& det_nodes,
      const std::vector<Node*>& sto_nodes) {
    g->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);
    uint K = tgt_node->value._matrix.size();
    // Cast needed to access fields such as unconstrained_value:
    auto src_node = static_cast<oper::StochasticOperator*>(tgt_node);
    // @lint-ignore CLANGTIDY
    auto dirichlet_distribution_node = src_node->in_nodes[0];
    auto param_node = dirichlet_distribution_node->in_nodes[0];
    for (uint k = 0; k < K; k++) {
      // Prepare gradients
      // Grad1 = (dY_1/dX_k, dY_2/dX_k, ..., dY_K/X_k)
      // where dY_k/dX_k = (sum(X) - X_k)/sum(X)^2
      //       dY_j/dX_k = - X_j/sum(X)^2, for j != k
      // Grad2 = (d^2Y_1/dX^2_k, ..., d^2Y_K/X^2_k)
      // where d2Y_k/dX2_k = -2 * (sum(X) - X_k)/sum(X)^3
      //       d2Y_j/dX2_k = -2 * X_j/sum(X)^3
      double param_a = param_node->value._matrix.coeff(k);
      double old_X_k = src_node->unconstrained_value._matrix.coeff(k);
      double sum = src_node->unconstrained_value._matrix.sum();
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
      double old_sto_descendants_log_prob;
      auto old_prop = create_proposer_dirichlet_gamma(
          sto_nodes,
          tgt_node,
          param_a,
          old_value,
          /* out */ old_sto_descendants_log_prob);

      NodeValue new_value = sample(old_prop);

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

      double new_sto_descendants_log_prob;
      auto new_prop = create_proposer_dirichlet_gamma(
          sto_nodes,
          tgt_node,
          param_a,
          new_value,
          /* out */ new_sto_descendants_log_prob);
      double logacc = new_sto_descendants_log_prob -
          old_sto_descendants_log_prob + new_prop->log_prob(old_value) -
          old_prop->log_prob(new_value);
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

  /*
  We treat the 2-dimensional Dirichlet sample as a Beta sample
  i.e. Let X_1 ~ Beta(alpha, beta)
  Y_1 = X_1 and Y_2 = 1 - X_1
  We store Y in the attribute value, and X in unconstrainted_value.
  */
  void nmc_step_for_dirichlet_beta(
      Node* tgt_node,
      const std::vector<Node*>& det_nodes,
      const std::vector<Node*>& sto_nodes) {
    g->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);
    assert(tgt_node->value._matrix.size() == 2);
    auto src_node = static_cast<oper::StochasticOperator*>(tgt_node);
    // @lint-ignore CLANGTIDY
    auto param_a = src_node->in_nodes[0]->in_nodes[0]->value._matrix.coeff(0);
    auto param_b = src_node->in_nodes[0]->in_nodes[0]->value._matrix.coeff(1);
    double old_X_k;
    // Prepare gradients
    // Grad1 = (dY_1/dX_1, dY_2/dX_1)
    // where dY_1/dX_1 = 1
    //       dY_j/dX_k = -1
    // Grad2 = (d^2Y_1/dX^2_1, d^2Y_2/X^2_1)
    // where d2Y_k/dX2_k = 0
    //       d2Y_j/dX2_k = 0
    old_X_k = src_node->value._matrix.coeff(0);
    Eigen::MatrixXd Grad1(2, 1);
    Grad1 << 1, -1;
    src_node->Grad1 = Grad1;
    *(src_node->Grad1.data() + 1) = -1;
    src_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
    src_node->grad1 = 1;
    src_node->grad2 = 0;

    // Propagate gradients
    NodeValue old_value(AtomicType::PROBABILITY, old_X_k);
    save_old_values(det_nodes);
    compute_gradients(det_nodes);
    double old_sto_descendants_log_prob;
    auto old_prop = create_proposer_dirichlet_beta(
        sto_nodes,
        tgt_node,
        param_a,
        param_b,
        old_value,
        /* out */ old_sto_descendants_log_prob);

    NodeValue new_value = sample(old_prop);
    *(src_node->value._matrix.data()) = new_value._double;
    *(src_node->value._matrix.data() + 1) = 1 - new_value._double;

    src_node->Grad1 = Grad1;
    src_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
    eval(det_nodes);
    compute_gradients(det_nodes);

    double new_sto_descendants_log_prob;
    auto new_prop = create_proposer_dirichlet_beta(
        sto_nodes,
        tgt_node,
        param_a,
        param_b,
        new_value,
        /* out */ new_sto_descendants_log_prob);
    double logacc = new_sto_descendants_log_prob -
        old_sto_descendants_log_prob + new_prop->log_prob(old_value) -
        old_prop->log_prob(new_value);
    // Accept or reject, reset (values and) gradients
    bool accepted = logacc > 0 or util::sample_logprob(gen, logacc);
    if (!accepted) {
      restore_old_values(det_nodes);
      *(src_node->value._matrix.data()) = old_X_k;
      *(src_node->value._matrix.data() + 1) = 1 - old_X_k;
    }
    clear_gradients(det_nodes);
    tgt_node->grad1 = 0;
    tgt_node->grad2 = 0;
    g->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
  }
};

void Graph::nmc(uint num_samples, std::mt19937& gen) {
  Graph::NMC(this, gen).infer(num_samples);
}

} // namespace graph
} // namespace beanmachine
