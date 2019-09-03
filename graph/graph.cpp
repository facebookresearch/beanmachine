// Copyright (c) Facebook, Inc. and its affiliates.
#include <random>
#include <sstream>

#include <folly/String.h>

#include "distribution.h"
#include "graph.h"
#include "operator.h"

namespace beanmachine {
namespace graph {

bool Node::is_stochastic() const {
  return (
      node_type == NodeType::OPERATOR and
      static_cast<const oper::Operator*>(this)->op_type ==
          OperatorType::SAMPLE);
}

double Node::log_prob() const {
  assert(is_stochastic());
  return static_cast<const distribution::Distribution*>(in_nodes[0])
      ->log_prob(value);
}

std::string Graph::to_string() {
  std::ostringstream os;
  for (auto const& node : nodes) {
    os << "Node " << node->index << " type "
       << static_cast<int>(node->node_type) << " parents [ ";
    for (Node* parent : node->in_nodes) {
      os << parent->index << " ";
    }
    os << "] children [ ";
    for (Node* child : node->out_nodes) {
      os << child->index << " ";
    }
    os << "]";
    if (node->value.type == AtomicType::UNKNOWN) {
      os << " unknown value "; // this is not an error, e.g. distribution node
    } else if (node->value.type == AtomicType::BOOLEAN) {
      os << " boolean value " << node->value._bool;
    } else if (node->value.type == AtomicType::REAL) {
      os << " real value " << node->value._double;
    } else if (node->value.type == AtomicType::TENSOR) {
      os << " tensor value " << node->value._tensor;
    } else {
      os << " BAD value";
    }
    os << std::endl;
  }
  return os.str();
}

std::vector<Node*> Graph::convert_parent_ids(
    const std::vector<uint>& parent_ids) const {
  // check that the parent ids are valid indices and convert them to
  // an array of Node* pointers
  std::vector<Node*> parent_nodes;
  for (uint paridx : parent_ids) {
    if (paridx >= nodes.size()) {
      throw std::out_of_range(folly::stringPrintf(
          "parent node_id (%u) must be less than %lu", paridx, nodes.size()));
    }
    parent_nodes.push_back(nodes[paridx].get());
  }
  return parent_nodes;
}

uint Graph::add_node(std::unique_ptr<Node> node, std::vector<uint> parents) {
  for (uint paridx : parents) {
    Node* parent = nodes[paridx].get();
    parent->out_nodes.push_back(node.get());
    node->in_nodes.push_back(parent);
  }
  uint index = node->index = nodes.size();
  nodes.push_back(std::move(node));
  return index;
}

Node* Graph::check_node(uint node_id, NodeType node_type) {
  if (node_id >= nodes.size()) {
    throw std::out_of_range(folly::stringPrintf(
        "node_id (%u) must be less than %lu", node_id, nodes.size()));
  }
  Node* node = nodes[node_id].get();
  if (node->node_type != node_type) {
    throw std::invalid_argument(folly::stringPrintf(
        "node_id %u expected type %d but actual type %d",
        node_id,
        static_cast<int>(node_type),
        static_cast<int>(node->node_type)));
  }
  return node;
}

uint Graph::add_constant(bool value) {
  return add_constant(AtomicValue(value));
}

uint Graph::add_constant(double value) {
  return add_constant(AtomicValue(value));
}

uint Graph::add_constant(torch::Tensor value) {
  return add_constant(AtomicValue(value));
}

uint Graph::add_constant(AtomicValue value) {
  std::unique_ptr<ConstNode> node = std::make_unique<ConstNode>(value);
  // constants don't have parents
  return add_node(std::move(node), std::vector<uint>());
}

uint Graph::add_distribution(
    DistributionType dist_type,
    AtomicType sample_type,
    std::vector<uint> parent_ids) {
  std::vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  // create a distribution node
  std::unique_ptr<Node> node = distribution::Distribution::new_distribution(
      dist_type, sample_type, parent_nodes);
  // and add the node to the graph
  return add_node(std::move(node), parent_ids);
}

uint Graph::add_operator(OperatorType op_type, std::vector<uint> parent_ids) {
  std::vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  std::unique_ptr<Node> node =
      std::make_unique<oper::Operator>(op_type, parent_nodes);
  return add_node(std::move(node), parent_ids);
}

void Graph::observe(uint node_id, bool val) {
  observe(node_id, AtomicValue(val));
}

void Graph::observe(uint node_id, double val) {
  observe(node_id, AtomicValue(val));
}

void Graph::observe(uint node_id, torch::Tensor val) {
  observe(node_id, AtomicValue(val));
}

void Graph::observe(uint node_id, AtomicValue value) {
  if (value.type != AtomicType::BOOLEAN) {
    throw std::invalid_argument("observe expects a boolean value");
  }
  Node* node = check_node(node_id, NodeType::OPERATOR);
  oper::Operator* op = static_cast<oper::Operator*>(node);
  if (op->op_type != OperatorType::SAMPLE) {
    throw std::invalid_argument("only sample nodes may be observed");
  }
  if (observed.find(node_id) != observed.end()) {
    throw std::invalid_argument(
        folly::stringPrintf("duplicate observe for node_id %u", node_id));
  }
  node->value = value;
  observed.insert(node_id);
}

uint Graph::query(uint node_id) {
  check_node(node_id, NodeType::OPERATOR);
  if (queried.find(node_id) != queried.end()) {
    throw std::invalid_argument(
        folly::stringPrintf("duplicate query for node_id %u", node_id));
  }
  queries.push_back(node_id);
  queried.insert(node_id);
  return queries.size() - 1; // the index is 0-based
}

void Graph::collect_sample() {
  // construct a sample of the queried nodes
  std::vector<AtomicValue> sample;
  for (uint node_id : queries) {
    sample.push_back(nodes[node_id]->value);
  }
  samples.push_back(sample);
}

std::vector<std::vector<AtomicValue>>&
Graph::infer(uint num_samples, InferenceType algorithm, uint seed) {
  if (queries.size() == 0) {
    throw std::runtime_error("no nodes queried for inference");
  }
  std::mt19937 generator(seed);
  samples.clear();
  if (algorithm == InferenceType::REJECTION) {
    rejection(num_samples, generator);
  } else if (algorithm == InferenceType::GIBBS) {
    gibbs(num_samples, generator);
  }
  return samples;
}

} // namespace graph
} // namespace beanmachine
