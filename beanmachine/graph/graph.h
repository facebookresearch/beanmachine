// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#ifndef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif

namespace beanmachine {
namespace graph {

const double PRECISION = 1e-10; // minimum precision of values

enum class AtomicType {
  UNKNOWN = 0,
  BOOLEAN = 1,
  PROBABILITY,
  REAL,
  POS_REAL, // Real numbers greater than *or* equal to zero
  NATURAL, // note: NATURAL numbers include zero (ISO 80000-2)
  TENSOR };

typedef unsigned long long int natural_t;

class AtomicValue {
 public:
  AtomicType type;
  union {
    bool _bool;
    double _double;
    natural_t _natural;
  };
  torch::Tensor _tensor;
  AtomicValue() : type(AtomicType::UNKNOWN) {}
  explicit AtomicValue(bool value) : type(AtomicType::BOOLEAN), _bool(value) {}
  explicit AtomicValue(double value) : type(AtomicType::REAL), _double(value) {}
  explicit AtomicValue(natural_t value) : type(AtomicType::NATURAL), _natural(value) {}
  explicit AtomicValue(torch::Tensor value)
      : type(AtomicType::TENSOR), _tensor(value.clone()) {}
  AtomicValue(AtomicType type, double value);
  bool operator==(const AtomicValue& other) const {
    return type == other.type and
        ((type == AtomicType::BOOLEAN and _bool == other._bool) or
         (type == AtomicType::REAL and _double == other._double) or
         (type == AtomicType::POS_REAL and _double == other._double) or
         (type == AtomicType::PROBABILITY and _double == other._double) or
         (type == AtomicType::NATURAL and _natural == other._natural) or
         (type == AtomicType::TENSOR and
          _tensor.eq(other._tensor).all().item<uint8_t>()));
  }
  bool operator!=(const AtomicValue& other) const {
    return not(*this == other);
  }
};

enum class OperatorType {
  UNKNOWN = 0,
  SAMPLE = 1, // This is the ~ operator in models
  TO_REAL,
  TO_POS_REAL,
  TO_TENSOR,
  COMPLEMENT,
  NEGATE,
  EXP,
  EXPM1,
  MULTIPLY,
  ADD,
};

enum class DistributionType {
  UNKNOWN = 0,
  TABULAR = 1,
  BERNOULLI = 2,
  BERNOULLI_NOISY_OR = 3,
  BETA = 4,
  BINOMIAL = 5,
  FLAT = 6,
  NORMAL = 7,
  HALF_CAUCHY = 8,
};

enum class NodeType {
  UNKNOWN = 0,
  CONSTANT = 1,
  DISTRIBUTION = 2,
  OPERATOR = 3,
  MAX
};

enum class InferenceType { UNKNOWN = 0, REJECTION = 1, GIBBS, NMC };

enum class AggregationType { UNKNOWN = 0, NONE = 1, MEAN};

class Node {
 public:
  bool is_observed = false;
  NodeType node_type;
  uint index; // index in Graph::nodes
  std::vector<Node*> in_nodes;
  std::vector<Node*> out_nodes;
  std::vector<uint> det_anc; // deterministic (operator) ancestors
  std::vector<uint> sto_anc; // stochastic ancestors
  AtomicValue value;
  double grad1;
  double grad2;
  bool is_stochastic() const;
  double log_prob() const; // only valid for stochastic nodes
  // gradient_log_prob is also only valid for stochastic nodes
  // this function adds the gradients to the passed in gradients
  void gradient_log_prob(double& grad1, double& grad2) const;
  Node() {}
  explicit Node(NodeType node_type) : node_type(node_type), grad1(0), grad2(0) {}
  Node(NodeType node_type, AtomicValue value)
      : node_type(node_type), value(value), grad1(0), grad2(0) {}
  // evaluate the node and store the result in `value` if appropriate
  // eval may involve sampling and that's why we need the random number engine
  virtual void eval(std::mt19937& gen) = 0;
  // populate the grad1 and grad2 fields
  virtual void compute_gradients() {}
  virtual ~Node() {}
};

class ConstNode : public Node {
 public:
  explicit ConstNode(AtomicValue value) : Node(NodeType::CONSTANT, value) {}
  void eval(std::mt19937& /* unused */) override {}
  ~ConstNode() override {}
};

// NOTE: the second kind of node -- Distribution is defined in distribution.h
// NOTE: the third kind of node -- Operator is defined in operator.h

struct Graph {
  Graph() {}
  ~Graph() {}
  std::string to_string();
  // Graph builder APIs -> return the node number
  uint add_constant(bool value);
  uint add_constant(double value);
  uint add_constant(natural_t value);
  uint add_constant(torch::Tensor value);
  uint add_constant(AtomicValue value);
  uint add_constant_probability(double value);
  uint add_constant_pos_real(double value);
  uint add_distribution(
      DistributionType dist_type,
      AtomicType sample_type,
      std::vector<uint> parents);
  uint add_operator(OperatorType op, std::vector<uint> parents);
  // inference related
  void observe(uint var, bool val);
  void observe(uint var, double val);
  void observe(uint var, natural_t val);
  void observe(uint var, torch::Tensor val);
  void observe(uint var, AtomicValue val);
  uint query(uint var); // returns the index of the query in the samples
  std::vector<std::vector<AtomicValue>>&
  infer(uint num_samples, InferenceType algorithm, uint seed = 5123401);
  std::vector<double>&
  infer_mean(uint num_samples, InferenceType algorithm, uint seed = 5123401);
  /*
  Use mean-field variational inference to infer the posterior mean, variance
  of the queried nodes in the graph.

  :param num_iters: The number of iterations to improve upon the estimates.
  :param steps_per_iter: The number of samples generated to make the estimate
                         in each iteration for each node.
  :param seed: The random number generator seed (default: 5123401)
  :param elbo_samples: The number of Monte Carlo samples to estimate the
                       ELBO (Evidence Lower Bound). Default 0 => no estimate.
  :returns: vector of parameters for each queried node;
            each parameter is itself a vector whose length depends
            on the type of the queried node
  :raises: std::runtime_error, std::invalid_argument
  */
  std::vector<std::vector<double>>&
  variational(uint num_iters, uint steps_per_iter, uint seed = 5123401,
    uint elbo_samples=0);
  std::vector<double>& get_elbo() {return elbo_vals;}
  std::set<uint> compute_support();
  std::tuple<std::vector<uint>, std::vector<uint>> compute_descendants(
      uint node_id, const std::set<uint> &support);
  std::tuple<std::vector<uint>, std::vector<uint>> compute_ancestors(
      uint node_id);
  /*
  Evaluate the target node and compute its gradient w.r.t. source_node
  (used for unit tests)
  :param tgt_idx: The index of the node to eval and compute grads.
  :param src_idx: The index of the node to compute the gradient w.r.t.
  :param seed: Random number generator seed.
  :param value: Output value of target node.
  :param grad1: Output value of first gradient.
  :param grad2: Output value of second gradient.
  */
  void eval_and_grad(
    uint tgt_idx, uint src_idx, uint seed, AtomicValue& value, double& grad1, double& grad2);
  /*
  Evaluate the deterministic descendants of the source node and compute
  the logprob_gradient of all stochastic descendants in the support including
  the source node.
  :param src_idx: The index of the node to evaluate the gradients w.r.t.
  :param grad1: Output value of first gradient.
  :param grad2: Output value of second gradient.
  */
  void gradient_log_prob(uint src_idx, double& grad1, double& grad2);
  /*
  Evaluate the deterministic descendants of the source node and compute
  the sum of logprob of all stochastic descendants in the support including
  the source node.
  :param src_idx: source node
  :returns: The sum of log_prob of source node and all stochastic descendants.
  */
  double log_prob(uint src_idx);

 private:
  uint add_node(std::unique_ptr<Node> node, std::vector<uint> parents);
  std::vector<Node*> convert_parent_ids(const std::vector<uint>& parents) const;
  Node* check_node(uint node_id, NodeType node_type);
  void _infer(uint num_samples, InferenceType algorithm, uint seed);
  std::vector<std::unique_ptr<Node>> nodes; // all nodes in topological order
  std::set<uint> observed; // set of observed nodes
  // we store redundant information in queries and queried. The latter is a
  // cache of the queried nodes while the former gives the order of nodes
  // queried
  std::vector<uint> queries; // list of queried nodenums
  std::set<uint> queried; // set of queried nodes
  std::vector<std::vector<AtomicValue>> samples;
  std::vector<double> means;
  AggregationType agg_type;
  uint agg_samples;
  std::vector<std::vector<double>> variational_params;
  std::vector<double> elbo_vals;
  void collect_sample();
  void rejection(uint num_samples, std::mt19937& gen);
  void gibbs(uint num_samples, std::mt19937& gen);
  void nmc(uint num_samples, std::mt19937& gen);
  void cavi(
    uint num_iters, uint steps_per_iter, std::mt19937& gen, uint elbo_samples);
};

} // namespace graph
} // namespace beanmachine
