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
#include <torch/torch.h>

namespace beanmachine {
namespace graph {

enum class AtomicType { UNKNOWN = 0, BOOLEAN = 1, REAL, TENSOR };

class AtomicValue {
 public:
  AtomicType type;
  union {
    bool _bool;
    double _double;
  };
  torch::Tensor _tensor;
  AtomicValue() : type(AtomicType::UNKNOWN) {}
  explicit AtomicValue(bool value) : type(AtomicType::BOOLEAN), _bool(value) {}
  explicit AtomicValue(double value) : type(AtomicType::REAL), _double(value) {}
  explicit AtomicValue(torch::Tensor value)
      : type(AtomicType::TENSOR), _tensor(value.clone()) {}
  bool operator==(const AtomicValue& other) const {
    return type == other.type and
        ((type == AtomicType::BOOLEAN and _bool == other._bool) or
         (type == AtomicType::REAL and _double == other._double) or
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
  NEGATE,
  EXP,
  MULTIPLY,
  ADD,
};

enum class DistributionType { UNKNOWN = 0, TABULAR = 1, BERNOULLI };

enum class NodeType {
  UNKNOWN = 0,
  CONSTANT = 1,
  DISTRIBUTION = 2,
  OPERATOR = 3,
  MAX
};

enum class InferenceType { UNKNOWN = 0, REJECTION = 1, GIBBS };

class Node {
 public:
  NodeType node_type;
  uint index; // index in Graph::nodes
  std::vector<Node*> in_nodes;
  std::vector<Node*> out_nodes;
  AtomicValue value;
  bool is_stochastic() const;
  double log_prob() const; // only valid for stochastic nodes
  Node() {}
  explicit Node(NodeType node_type) : node_type(node_type) {}
  Node(NodeType node_type, AtomicValue value)
      : node_type(node_type), value(value) {}
  // evaluate the node and store the result in `value` if appropriate
  // eval may involve sampling and that's why we need the random number engine
  virtual void eval(std::mt19937& gen) = 0;
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
  uint add_constant(torch::Tensor value);
  uint add_constant(AtomicValue value);
  uint add_distribution(
      DistributionType dist_type,
      AtomicType sample_type,
      std::vector<uint> parents);
  uint add_operator(OperatorType op, std::vector<uint> parents);
  // inference related
  void observe(uint var, bool val);
  void observe(uint var, double val);
  void observe(uint var, torch::Tensor val);
  void observe(uint var, AtomicValue val);
  uint query(uint var); // returns the index of the query in the samples
  std::vector<std::vector<AtomicValue>>&
  infer(uint num_samples, InferenceType algorithm, uint seed = 5123401);
  std::set<uint> compute_support();
  std::tuple<std::list<uint>, std::list<uint>> compute_descendants(
      uint node_id);

 private:
  uint add_node(std::unique_ptr<Node> node, std::vector<uint> parents);
  std::vector<Node*> convert_parent_ids(const std::vector<uint>& parents) const;
  Node* check_node(uint node_id, NodeType node_type);
  std::vector<std::unique_ptr<Node>> nodes; // all nodes in topological order
  std::set<uint> observed; // set of observed nodes
  // we store redundant information in queries and queried. The latter is a
  // cache of the queried nodes while the former gives the order of nodes
  // queried
  std::vector<uint> queries; // list of queried nodenums
  std::set<uint> queried; // set of queried nodes
  std::vector<std::vector<AtomicValue>> samples;
  void collect_sample();
  void rejection(uint num_samples, std::mt19937& gen);
  void gibbs(uint num_samples, std::mt19937& gen);
};

} // namespace graph
} // namespace beanmachine
