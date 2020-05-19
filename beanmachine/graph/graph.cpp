// Copyright (c) Facebook, Inc. and its affiliates.
#include <random>
#include <sstream>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace graph {

AtomicValue::AtomicValue(AtomicType type, double value)
    : type(type), _double(value) {
  // don't allow constrained values to get too close to the boundary
  if (type == AtomicType::POS_REAL) {
    if (_double < PRECISION) {
      _double = PRECISION;
    }
  } else if (type == AtomicType::PROBABILITY) {
    if (_double < PRECISION) {
      _double = PRECISION;
    } else if (_double > (1 - PRECISION)) {
      _double = 1 - PRECISION;
    }
  } else {
    // this API is only meant for POS_REAL, REAL and PROBABILITY values
    if (type != AtomicType::REAL) {
      throw std::invalid_argument(
          "expect probability, pos_real, or real type with floating point value");
    }
  }
}

AtomicValue::AtomicValue(AtomicType type) : type(type) {
  switch (type) {
    case AtomicType::UNKNOWN:
      break;
    case AtomicType::BOOLEAN:
      _bool = false;
      break;
    case AtomicType::PROBABILITY:
    case AtomicType::REAL:
    case AtomicType::POS_REAL:
      _double = 0.0;
      break;
    case AtomicType::NATURAL:
      _natural = 0;
      break;
    case AtomicType::TENSOR:
      _tensor = torch::Tensor();
      break;
  }
}

std::string AtomicValue::to_string() const {
  std::ostringstream os;
  if (type == AtomicType::UNKNOWN) {
    os << "unknown value "; // this is not an error, e.g. distribution node
  } else if (type == AtomicType::BOOLEAN) {
    os << "boolean value " << _bool;
  } else if (type == AtomicType::PROBABILITY) {
    os << "probability value " << _double;
  } else if (type == AtomicType::REAL) {
    os << "real value " << _double;
  } else if (type == AtomicType::POS_REAL) {
    os << "pos real value " << _double;
  } else if (type == AtomicType::NATURAL) {
    os << "natural value " << _natural;
  } else if (type == AtomicType::TENSOR) {
    os << "tensor value " << _tensor;
  } else {
    os << "BAD value";
  }
  return os.str();
}

std::string Graph::to_string() const {
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
    os << "] " << node->value.to_string() << std::endl;
  }
  return os.str();
}

void Graph::eval_and_grad(
    uint tgt_idx,
    uint src_idx,
    uint seed,
    AtomicValue& value,
    double& grad1,
    double& grad2) {
  if (src_idx >= nodes.size()) {
    throw std::out_of_range("src_idx " + std::to_string(src_idx));
  }
  if (tgt_idx >= nodes.size() or tgt_idx <= src_idx) {
    throw std::out_of_range("tgt_idx " + std::to_string(tgt_idx));
  }
  // initialize the gradients of the source node to get the computation started
  Node* src_node = nodes[src_idx].get();
  src_node->grad1 = 1;
  src_node->grad2 = 0;
  std::mt19937 generator(seed);
  for (uint node_id = src_idx + 1; node_id <= tgt_idx; node_id++) {
    Node* node = nodes[node_id].get();
    node->eval(generator);
    node->compute_gradients();
    if (node->index == tgt_idx) {
      value = node->value;
      grad1 = node->grad1;
      grad2 = node->grad2;
    }
  }
  // reset all the gradients including the source node
  for (uint node_id = src_idx; node_id <= tgt_idx; node_id++) {
    Node* node = nodes[node_id].get();
    node->grad1 = node->grad2 = 0;
  }
}

void Graph::gradient_log_prob(uint src_idx, double& grad1, double& grad2) {
  Node* src_node = check_node(src_idx, NodeType::OPERATOR);
  if (not src_node->is_stochastic()) {
    throw std::runtime_error(
        "gradient_log_prob only supported on stochastic nodes");
  }
  // start gradient
  src_node->grad1 = 1;
  src_node->grad2 = 0;
  std::mt19937 generator(12131); // seed is irrelevant for deterministic ops
  auto supp = compute_support();
  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) = compute_descendants(src_idx, supp);
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    node->eval(generator);
    node->compute_gradients();
  }
  grad1 = grad2 = 0;
  for (auto node_id : sto_nodes) {
    Node* node = nodes[node_id].get();
    node->gradient_log_prob(grad1, grad2);
  }
  src_node->grad1 = 0; // end gradient computation reset grads
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    node->grad1 = node->grad2 = 0;
  }
}

double Graph::log_prob(uint src_idx) {
  Node* src_node = check_node(src_idx, NodeType::OPERATOR);
  if (not src_node->is_stochastic()) {
    throw std::runtime_error("log_prob only supported on stochastic nodes");
  }
  std::mt19937 generator(12131); // seed is irrelevant for deterministic ops
  auto supp = compute_support();
  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) = compute_descendants(src_idx, supp);
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    node->eval(generator);
  }
  double log_prob = 0.0;
  for (auto node_id : sto_nodes) {
    Node* node = nodes[node_id].get();
    log_prob += node->log_prob();
  }
  return log_prob;
}

std::vector<Node*> Graph::convert_parent_ids(
    const std::vector<uint>& parent_ids) const {
  // check that the parent ids are valid indices and convert them to
  // an array of Node* pointers
  std::vector<Node*> parent_nodes;
  for (uint paridx : parent_ids) {
    if (paridx >= nodes.size()) {
      throw std::out_of_range(
          "parent node_id " + std::to_string(paridx) + "must be less than " +
          std::to_string(nodes.size()));
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
    } else {
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
        "node_id (" + std::to_string(node_id) + ") must be less than " +
        std::to_string(nodes.size()));
  }
  Node* node = nodes[node_id].get();
  if (node->node_type != node_type) {
    throw std::invalid_argument(
        "node_id " + std::to_string(node_id) + "expected type " +
        std::to_string(static_cast<int>(node_type)) + " but actual type " +
        std::to_string(static_cast<int>(node->node_type)));
  }
  return node;
}

uint Graph::add_constant(bool value) {
  return add_constant(AtomicValue(value));
}

uint Graph::add_constant(double value) {
  return add_constant(AtomicValue(value));
}

uint Graph::add_constant(natural_t value) {
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

uint Graph::add_constant_probability(double value) {
  if (value < 0 or value > 1) {
    throw std::invalid_argument("probability must be between 0 and 1");
  }
  return add_constant(AtomicValue(AtomicType::PROBABILITY, value));
}

uint Graph::add_constant_pos_real(double value) {
  if (value < 0) {
    throw std::invalid_argument("pos_real must be >=0");
  }
  return add_constant(AtomicValue(AtomicType::POS_REAL, value));
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

uint Graph::add_factor(FactorType fac_type, std::vector<uint> parent_ids) {
  std::vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  std::unique_ptr<Node> node =
      factor::Factor::new_factor(fac_type, parent_nodes);
  uint node_id = add_node(std::move(node), parent_ids);
  // factors are both stochastic nodes and observed nodes
  Node* node2 = check_node(node_id, NodeType::FACTOR);
  node2->is_observed = true;
  observed.insert(node_id);
  return node_id;
}

void Graph::observe(uint node_id, bool val) {
  observe(node_id, AtomicValue(val));
}

void Graph::observe(uint node_id, double val) {
  Node* node = check_node(node_id, NodeType::OPERATOR);
  observe(node_id, AtomicValue(node->value.type, val));
}

void Graph::observe(uint node_id, natural_t val) {
  observe(node_id, AtomicValue(val));
}

void Graph::observe(uint node_id, torch::Tensor val) {
  observe(node_id, AtomicValue(val));
}

void Graph::observe(uint node_id, AtomicValue value) {
  Node* node = check_node(node_id, NodeType::OPERATOR);
  oper::Operator* op = static_cast<oper::Operator*>(node);
  if (op->op_type != OperatorType::SAMPLE) {
    throw std::invalid_argument("only sample nodes may be observed");
  }
  if (observed.find(node_id) != observed.end()) {
    throw std::invalid_argument(
        "duplicate observe for node_id " + std::to_string(node_id));
  }
  if (node->value.type != value.type) {
    throw std::invalid_argument(
        "observe expected type " +
        std::to_string(static_cast<int>(node->value.type)) + " instead got " +
        std::to_string(static_cast<int>(value.type)));
  }
  node->value = value;
  node->is_observed = true;
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
  // note: we divide each new value by agg_samples rather than directly add
  // them to the total to avoid overflow
  else if (agg_type == AggregationType::MEAN) {
    uint pos = 0;
    for (uint node_id : queries) {
      AtomicValue value = nodes[node_id]->value;
      if (value.type == AtomicType::BOOLEAN) {
        means[pos] += double(value._bool) / agg_samples;
      } else if (
          value.type == AtomicType::REAL or
          value.type == AtomicType::POS_REAL or
          value.type == AtomicType::PROBABILITY) {
        means[pos] += value._double / agg_samples;
      } else if (value.type == AtomicType::NATURAL) {
        means[pos] += double(value._natural) / agg_samples;
      } else {
        throw std::runtime_error(
            "Mean aggregation only supported for "
            "boolean/real/probability/natural-valued nodes");
      }
      pos++;
    }
  } else {
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
  } else if (algorithm == InferenceType::NMC) {
    nmc(num_samples, generator);
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
  agg_samples = num_samples;
  means.clear();
  means.resize(queries.size(), 0.0);
  _infer(num_samples, algorithm, seed);
  return means;
}

std::vector<std::vector<double>>& Graph::variational(
    uint num_iters,
    uint steps_per_iter,
    uint seed,
    uint elbo_samples) {
  if (queries.size() == 0) {
    throw std::runtime_error("no nodes queried for inference");
  }
  for (uint node_id : queries) {
    Node* node = nodes[node_id].get();
    if (not node->is_stochastic()) {
      throw std::invalid_argument(
          "only sample nodes may be queried in "
          "variational inference");
    }
  }
  elbo_vals.clear();
  std::mt19937 generator(seed);
  cavi(num_iters, steps_per_iter, generator, elbo_samples);
  return variational_params;
}

std::vector<uint> Graph::get_parent_ids(const std::vector<Node*>& parent_nodes) const {
  std::vector<uint> parent_ids;
  for (auto node : parent_nodes) {
    parent_ids.push_back(node->index);
  }
  return parent_ids;
}

Graph::Graph(const Graph& other) {
  // This copy constructor does not copy the inference results (if available)
  // from the source graph.
  for (uint i = 0; i < other.nodes.size(); i++) {
    Node* node = other.nodes[i].get();
    std::vector<uint> parent_ids = get_parent_ids(node->in_nodes);
    switch(node->node_type) {
      case NodeType::CONSTANT: {
        AtomicValue value_copy = AtomicValue(node->value);
        add_constant(value_copy);
        break;
      }
      case NodeType::DISTRIBUTION: {
        distribution::Distribution* dist = static_cast<distribution::Distribution*>(node);
        add_distribution(dist->dist_type, dist->sample_type, parent_ids);
        break;
      }
      case NodeType::OPERATOR: {
        add_operator(static_cast<oper::Operator*>(node)->op_type, parent_ids);
        if (node->is_observed) {
          observe(node->index, AtomicValue(node->value));
        }
        break;
      }
      case NodeType::FACTOR: {
        add_factor(static_cast<factor::Factor*>(node)->fac_type, parent_ids);
        break;
      }
      default: {
        throw std::invalid_argument("Trying to copy a node of unknown type.");
      }
    }
  }
  for (uint node_id : other.queries) {
    query(node_id);
  }
}

} // namespace graph
} // namespace beanmachine
