// Copyright (c) Facebook, Inc. and its affiliates.
#include <random>
#include <sstream>

#include "beanmachine/graph/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator.h"

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
      throw std::out_of_range(
          "parent node_id " + std::to_string(paridx)
          + "must be less than " + std::to_string(nodes.size()));
    }
    parent_nodes.push_back(nodes[paridx].get());
  }
  return parent_nodes;
}

uint Graph::add_node(std::unique_ptr<Node> node, std::vector<uint> parents) {
  // for each parent collect their deterministic/stochastic ancestors
  // also maintain the in/out nodes of this node and its parents
  std::set<uint> det_set;
  std::set<uint> sto_set;
  for (uint paridx : parents) {
    Node* parent = nodes[paridx].get();
    parent->out_nodes.push_back(node.get());
    node->in_nodes.push_back(parent);
    if (parent->is_stochastic()) {
      sto_set.insert(parent->index);
    }
    else {
      det_set.insert(parent->det_anc.begin(), parent->det_anc.end());
      if (parent->node_type == NodeType::OPERATOR) {
        det_set.insert(parent->index);
      }
      sto_set.insert(parent->sto_anc.begin(), parent->sto_anc.end());
    }
  }
  node->det_anc.insert(node->det_anc.end(), det_set.begin(), det_set.end());
  node->sto_anc.insert(node->sto_anc.end(), sto_set.begin(), sto_set.end());
  uint index = node->index = nodes.size();
  nodes.push_back(std::move(node));
  return index;
}

Node* Graph::check_node(uint node_id, NodeType node_type) {
  if (node_id >= nodes.size()) {
    throw std::out_of_range(
        "node_id (" + std::to_string(node_id) + ") must be less than "
        + std::to_string(nodes.size()));
  }
  Node* node = nodes[node_id].get();
  if (node->node_type != node_type) {
    throw std::invalid_argument(
      "node_id " + std::to_string(node_id) + "expected type "
      + std::to_string(static_cast<int>(node_type))
      + " but actual type "
      + std::to_string(static_cast<int>(node->node_type)));
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
      "duplicate observe for node_id " + std::to_string(node_id));
  }
  node->value = value;
  observed.insert(node_id);
}

uint Graph::query(uint node_id) {
  check_node(node_id, NodeType::OPERATOR);
  if (queried.find(node_id) != queried.end()) {
    throw std::invalid_argument(
      "duplicate query for node_id " + std::to_string(node_id));
  }
  queries.push_back(node_id);
  queried.insert(node_id);
  return queries.size() - 1; // the index is 0-based
}

void Graph::collect_sample() {
  // construct a sample of the queried nodes
  if (agg_type == AggregationType::NONE) {
    std::vector<AtomicValue> sample;
    for (uint node_id : queries) {
      sample.push_back(nodes[node_id]->value);
    }
    samples.push_back(sample);
  }
  else if (agg_type == AggregationType::MEAN) {
    uint pos = 0;
    for (uint node_id : queries) {
      AtomicValue value = nodes[node_id]->value;
      if (value.type == AtomicType::BOOLEAN) {
        means[pos] += double(value._bool);
      }
      else if (value.type == AtomicType::REAL) {
        means[pos] += value._double;
      }
      else {
        throw std::runtime_error("Mean aggregation only supported for "
          "boolean- and real-valued nodes");
      }
      pos++;
    }
  }
  else {
    assert(false);
  }
}

void Graph::_infer(uint num_samples, InferenceType algorithm, uint seed) {
  if (queries.size() == 0) {
    throw std::runtime_error("no nodes queried for inference");
  }
  if (num_samples < 1) {
    throw std::runtime_error("num_samples can't be zero");
  }
  std::mt19937 generator(seed);
  if (algorithm == InferenceType::REJECTION) {
    rejection(num_samples, generator);
  } else if (algorithm == InferenceType::GIBBS) {
    gibbs(num_samples, generator);
  }
}

std::vector<std::vector<AtomicValue>>&
Graph::infer(uint num_samples, InferenceType algorithm, uint seed) {
  agg_type = AggregationType::NONE;
  samples.clear();
  _infer(num_samples, algorithm, seed);
  return samples;
}

std::vector<double>&
Graph::infer_mean(uint num_samples, InferenceType algorithm, uint seed) {
  agg_type = AggregationType::MEAN;
  means.clear();
  means.resize(queries.size(), 0.0);
  _infer(num_samples, algorithm, seed);
  for (uint i=0; i<means.size(); i++) {
    means[i] /= num_samples;
  }
  return means;
}

std::vector<std::vector<double>>&
Graph::variational(uint num_iters, int steps_per_iter, uint seed) {
  if (queries.size() == 0) {
    throw std::runtime_error("no nodes queried for inference");
  }
  for (uint node_id : queries) {
    Node * node = nodes[node_id].get();
    if (not node->is_stochastic()) {
      throw std::invalid_argument("only sample nodes may be queried in "
        "variational inference");
    }
  }
  std::mt19937 generator(seed);
  cavi(num_iters, steps_per_iter, generator);
  return variational_params;
}

} // namespace graph
} // namespace beanmachine
