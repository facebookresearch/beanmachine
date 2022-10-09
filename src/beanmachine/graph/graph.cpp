/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>
// We must include cmath first thing with macro _USE_MATH_DEFINES
// to ensure the definition of math constants in Windows,
// before any other header files have the chance of including cmath
// without the macro.

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <variant>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/transform/transform.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

using namespace util;
using namespace std;

NATURAL_TYPE NATURAL_ZERO = 0ull;
NATURAL_TYPE NATURAL_ONE = 1ull;

string ValueType::to_string() const {
  string vtype;
  string atype;
  switch (atomic_type) {
    case AtomicType::UNKNOWN:
      atype = "unknown";
      break;
    case AtomicType::BOOLEAN:
      atype = "boolean";
      break;
    case AtomicType::PROBABILITY:
      atype = "probability";
      break;
    case AtomicType::REAL:
      atype = "real";
      break;
    case AtomicType::POS_REAL:
      atype = "positive real";
      break;
    case AtomicType::NEG_REAL:
      atype = "negative real";
      break;
    case AtomicType::NATURAL:
      atype = "natural";
      break;
  }
  switch (variable_type) {
    case VariableType::UNKNOWN:
      return "unknown variable";
    case VariableType::SCALAR:
      return atype;
    case VariableType::BROADCAST_MATRIX:
      vtype = "matrix<";
      break;
    case VariableType::COL_SIMPLEX_MATRIX:
      vtype = "col_simplex_matrix<";
      break;
  }
  return vtype + atype + ">";
}

NodeValue::NodeValue(AtomicType type, double value)
    : type(type), _double(value) {
  // don't allow constrained values to get too close to the boundary
  switch (type) {
    case AtomicType::POS_REAL:
      if (_double < PRECISION) {
        _double = PRECISION;
      }
      break;
    case AtomicType::NEG_REAL:
      if (_double > -PRECISION) {
        _double = -PRECISION;
      }
      break;
    case AtomicType::PROBABILITY:
      if (_double < PRECISION) {
        _double = PRECISION;
      } else if (_double > (1 - PRECISION)) {
        _double = 1 - PRECISION;
      }
      break;
    case AtomicType::REAL:
      break;
    default:
      // this API is only meant for POS_REAL, NEG_REAL, REAL and PROBABILITY
      // values
      throw invalid_argument(
          "expect probability, pos_real, neg_real or real type with floating point value");
  }
}

void NodeValue::init_scalar(AtomicType type) {
  switch (type) {
    case AtomicType::UNKNOWN:
      break;
    case AtomicType::BOOLEAN:
      _bool = false;
      break;
    case AtomicType::REAL:
      _double = 0.0;
      break;
    case AtomicType::PROBABILITY:
    case AtomicType::POS_REAL:
      _double = PRECISION;
      break;
    case AtomicType::NEG_REAL:
      _double = -PRECISION;
      break;
    case AtomicType::NATURAL:
      _natural = 0;
      break;
  }
}

NodeValue::NodeValue(AtomicType type) : type(type) {
  this->init_scalar(type);
}

NodeValue::NodeValue(ValueType type) : type(type) {
  if (type.variable_type == VariableType::BROADCAST_MATRIX) {
    switch (type.atomic_type) {
      case AtomicType::BOOLEAN:
        _bmatrix = Eigen::MatrixXb::Constant(type.rows, type.cols, false);
        break;
      case AtomicType::REAL:
        _matrix = Eigen::MatrixXd::Zero(type.rows, type.cols);
        break;
      case AtomicType::POS_REAL:
      case AtomicType::PROBABILITY:
        _matrix = Eigen::MatrixXd::Constant(type.rows, type.cols, PRECISION);
        break;
      case AtomicType::NEG_REAL:
        _matrix = Eigen::MatrixXd::Constant(type.rows, type.cols, -PRECISION);
        break;
      case AtomicType::NATURAL:
        _nmatrix =
            Eigen::MatrixXn::Constant(type.rows, type.cols, (natural_t)0);
        break;
      default:
        throw invalid_argument("Unsupported types for BROADCAST_MATRIX.");
    }
  } else if (type.variable_type == VariableType::COL_SIMPLEX_MATRIX) {
    _matrix = Eigen::MatrixXd::Ones(type.rows, type.cols) / type.rows;
  } else if (type.variable_type == VariableType::SCALAR) {
    this->init_scalar(type.atomic_type);
  } else {
    throw invalid_argument("Unsupported variable type.");
  }
}

string NodeValue::to_string() const {
  ostringstream os;
  string type_str = type.to_string() + " ";
  if (type.variable_type == VariableType::SCALAR) {
    switch (type.atomic_type) {
      case AtomicType::UNKNOWN:
        os << type_str;
        break;
      case AtomicType::BOOLEAN:
        os << type_str << _bool;
        break;
      case AtomicType::NATURAL:
        os << type_str << _natural;
        break;
      case AtomicType::REAL:
      case AtomicType::POS_REAL:
      case AtomicType::NEG_REAL:
      case AtomicType::PROBABILITY:
        os << type_str << _double;
        break;
      default:
        os << "Unsupported SCALAR value";
        break;
    }
  } else if (type.variable_type == VariableType::BROADCAST_MATRIX) {
    switch (type.atomic_type) {
      case AtomicType::UNKNOWN:
        os << type_str;
        break;
      case AtomicType::REAL:
      case AtomicType::POS_REAL:
      case AtomicType::NEG_REAL:
      case AtomicType::PROBABILITY:
        os << type_str << _matrix;
        break;
      case AtomicType::BOOLEAN:
        os << type_str << _bmatrix;
        break;
      case AtomicType::NATURAL:
        os << type_str << _nmatrix;
        break;

        break;
      default:
        os << "Unsupported BROADCAST_MATRIX value";
    }
  } else if (type.variable_type == VariableType::COL_SIMPLEX_MATRIX) {
    switch (type.atomic_type) {
      case AtomicType::UNKNOWN:
        os << type_str;
        break;
      case AtomicType::PROBABILITY:
        os << type_str << _matrix;
        break;
      default:
        os << "Unsupported COL_SIMPLEX_MATRIX value";
    }
  } else {
    os << "Unsupported NodeValue";
  }
  return os.str();
}

Node::Node(NodeType node_type, NodeValue value, const vector<Node*>& in_nodes)
    : node_type(node_type),
      in_nodes{in_nodes.begin(), in_nodes.end()},
      value(value),
      grad1(0),
      grad2(0) {}
// It might be tempting to set the `out_node` field in the `Node`
// constructor as well. However this would lead to a problem if the `Node`
// subclass' constructor threw an exception (for example because the in-nodes
// do not satisfy typing constraints), because then the node would be
// deallocated but a (now invalid) pointer to it would be kept in
// the `out_nodes` of its (former) in-nodes.
// One possible solution would be to remove those invalid pointers
// from in-nodes' `out_node` fields in in `~Node`, but that would in turn
// cause problems when `Graph` is destructed, because it might happen that the
// in-nodes of the current node are deallocated first, in which case
// attempting to access their `out_node` field causes a crash.
// In any case, it
// makes more sense to set `out_nodes` when the node is added to the graph
// because out-nodes do seem to be more a property of a *graph* than of a
// *node*, as opposed to in-nodes which are a more intrinsic part of a node.

// TODO: the following is used in beta.cpp only. Does it really need to be here?
// Why is it used there only, given the name sounds pretty generic?
// Are other classes using different versions of the same idea?
// Can it be de-duplicated?
template <class T1, class T2>
void Node::forward_gradient_scalarops(
    T1& jacobian,
    T2& hessian,
    double& d_grad1,
    double& d_grad2) const {
  uint in_degree = static_cast<uint>(in_nodes.size());
  assert(jacobian.cols() == in_degree);
  assert(hessian.cols() == in_degree and hessian.rows() == in_degree);

  T1 Grad1_old = T1::Zero();
  T1 Grad2_old = T1::Zero();
  for (uint i = 0; i < in_degree; i++) {
    *(Grad1_old.data() + i) = in_nodes[i]->grad1;
    *(Grad2_old.data() + i) = in_nodes[i]->grad2;
  }
  double grad1_update = jacobian * Grad1_old.transpose();
  double grad2_update =
      ((Grad1_old * hessian).array() * Grad1_old.array()).sum();
  grad2_update += jacobian * Grad2_old.transpose();

  d_grad1 += grad1_update;
  d_grad2 += grad2_update;
}

template void
Node::forward_gradient_scalarops<Eigen::Matrix<double, 1, 2>, Eigen::Matrix2d>(
    Eigen::Matrix<double, 1, 2>& jacobian,
    Eigen::Matrix2d& hessian,
    double& d_grad1,
    double& d_grad2) const;

void Node::reset_backgrad() {
  assert(value.type.variable_type != graph::VariableType::UNKNOWN);
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    back_grad1 = 0;
  } else {
    back_grad1.setZero(value.type.rows, value.type.cols);
  }
}

void Node::to_scalar() {
  switch (value.type.atomic_type) {
    case graph::AtomicType::BOOLEAN:
      assert(value._bmatrix.size() == 1);
      value._bool = *(value._bmatrix.data());
      value._bmatrix.setZero(0, 0);
      break;
    case graph::AtomicType::NATURAL:
      assert(value._nmatrix.size() == 1);
      value._natural = *(value._nmatrix.data());
      value._nmatrix.setZero(0, 0);
      break;
    case graph::AtomicType::REAL:
    case graph::AtomicType::POS_REAL:
    case graph::AtomicType::NEG_REAL:
    case graph::AtomicType::PROBABILITY:
      assert(value._matrix.size() == 1);
      value._double = *(value._matrix.data());
      value._matrix.setZero(0, 0);
      break;
    default:
      throw runtime_error("unsupported AtomicType to cast to scalar");
  }
}

Eigen::MatrixXd Node::log_prob_iid(const graph::NodeValue& value) const {
  Eigen::MatrixXd temp;
  log_prob_iid(value, temp);
  return temp; // fine because of move semantics
}

string Graph::to_string() const {
  ostringstream os;
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
    os << "] " << node->value.to_string() << endl;
  }
  return os.str();
}

void Graph::eval_and_update_backgrad(const vector<Node*>& mutable_support) {
  // generator doesn't matter for det nodes
  // TODO: add default generator
  mt19937 generator(12131);
  for (auto node : mutable_support) {
    node->reset_backgrad();
    if (!node->is_stochastic()) {
      node->eval(generator);
    }
  }

  for (auto it = mutable_support.rbegin(); it != mutable_support.rend(); ++it) {
    Node* node = *it;
    if (node->is_stochastic() and node->node_type == NodeType::OPERATOR) {
      auto sto_node = static_cast<oper::StochasticOperator*>(node);
      // TODO: Investigate the semantics of _backward(skip_observed),
      // understand when/why it is appropriate to pass true or false,,
      // and provide a correctness argument.
      sto_node->_backward(false);
      if (sto_node->transform_type != TransformType::NONE) {
        // sync value with unconstrained_value
        sto_node->get_original_value(true);
        sto_node->get_unconstrained_gradient();
      }
    } else {
      node->backward();
    }
  }
}

void Graph::eval_and_grad(
    NodeID tgt_idx,
    NodeID src_idx,
    NodeID seed,
    NodeValue& value,
    double& grad1,
    double& grad2) {
  // TODO: used for testing only, should integrate it with
  // whatever code is actually being used for eval and grad.
  if (src_idx >= static_cast<NodeID>(nodes.size())) {
    throw out_of_range("src_idx " + std::to_string(src_idx));
  }
  if (tgt_idx >= static_cast<NodeID>(nodes.size()) or tgt_idx <= src_idx) {
    throw out_of_range("tgt_idx " + std::to_string(tgt_idx));
  }
  // initialize the gradients of the source node to get the computation started
  Node* src_node = nodes[src_idx].get();
  src_node->grad1 = 1;
  src_node->grad2 = 0;
  mt19937 generator(seed);
  for (NodeID node_id = src_idx + 1; node_id <= tgt_idx; node_id++) {
    Node* node = nodes[node_id].get();
    if (node->is_mutable()) {
      node->eval(generator);
    }
    node->compute_gradients();
    if (node->index == tgt_idx) {
      value = node->value;
      grad1 = node->grad1;
      grad2 = node->grad2;
    }
  }
  // reset all the gradients including the source node
  for (NodeID node_id = src_idx; node_id <= tgt_idx; node_id++) {
    Node* node = nodes[node_id].get();
    node->grad1 = node->grad2 = 0;
  }
}

void Graph::_test_backgrad(
    MutableSupport& mutable_support,
    vector<DoubleMatrix*>& grad1) {
  for (auto it = mutable_support.begin(); it != mutable_support.end(); ++it) {
    Node* node = nodes[*it].get();
    node->reset_backgrad();
  }
  grad1.clear();
  for (auto it = mutable_support.rbegin(); it != mutable_support.rend(); ++it) {
    Node* node = nodes[*it].get();
    if (node->is_stochastic() and node->node_type == NodeType::OPERATOR) {
      auto sto_node = static_cast<oper::StochasticOperator*>(node);
      sto_node->_backward(false);
      if (sto_node->transform_type != TransformType::NONE) {
        sto_node->get_unconstrained_value(true);
        sto_node->get_unconstrained_gradient();
      }
      grad1.push_back(&node->back_grad1);
    } else {
      node->backward();
    }
  }
  reverse(grad1.begin(), grad1.end());
}

void Graph::test_grad(vector<DoubleMatrix*>& grad1) {
  auto mutable_support = compute_mutable_support();
  _test_backgrad(mutable_support, grad1);
}

void Graph::eval_and_grad(vector<DoubleMatrix*>& grad1, uint seed) {
  mt19937 generator(seed);
  auto mutable_support = compute_mutable_support();
  for (auto it = mutable_support.begin(); it != mutable_support.end(); ++it) {
    Node* node = nodes[*it].get();
    if (!node->is_observed) {
      node->eval(generator);
    }
  }
  _test_backgrad(mutable_support, grad1);
}

void set_value(Eigen::MatrixXd& variable, double value) {
  variable.setConstant(value);
}
void set_value(double& variable, double value) {
  variable = value;
}

void Graph::gradient_log_prob(NodeID src_idx, double& grad1, double& grad2) {
  // TODO: As of May 2021, this method is being used for testing only.
  // Refactor code so that we test the code actually being used for the
  // normal functionality of the class.
  // If that is not possible, this is an indication that this is code
  // is only useful for testing, and should therefore
  // be moved to the testing code.
  Node* src_node = check_node(src_idx, NodeType::OPERATOR);
  if (not src_node->is_stochastic()) {
    throw runtime_error("gradient_log_prob only supported on stochastic nodes");
  }
  src_node->grad1 = 1;
  src_node->grad2 = 0;

  auto mutable_support = compute_mutable_support();
  vector<NodeID> det_nodes;
  vector<NodeID> sto_nodes;
  tie(det_nodes, sto_nodes) = compute_affected_nodes(src_idx, mutable_support);
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    // passing generator for signature,
    // but it is irrelevant for deterministic nodes.
    // TODO: can we make signature use a default generator then?
    mt19937 generator(12131);
    node->eval(generator);
    node->compute_gradients();
  }
  set_value(grad1, 0.0);
  set_value(grad2, 0.0);
  for (auto node_id : sto_nodes) {
    Node* node = nodes[node_id].get();
    node->gradient_log_prob(src_node, grad1, grad2);
  }
  // TODO clarify why we need to reset gradients
  // if we seem to be computing them from scratch when needed.

  // end gradient computation reset grads
  src_node->grad1 = 0;
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    node->grad1 = node->grad2 = 0;
  }
}

double Graph::log_prob(NodeID src_idx) {
  // TODO: also used in tests only
  Node* src_node = check_node(src_idx, NodeType::OPERATOR);
  if (not src_node->is_stochastic()) {
    throw runtime_error("log_prob only supported on stochastic nodes");
  }
  auto mutable_support = compute_mutable_support();
  vector<NodeID> det_nodes;
  vector<NodeID> sto_nodes;
  tie(det_nodes, sto_nodes) = compute_affected_nodes(src_idx, mutable_support);
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    mt19937 generator(12131); // seed is irrelevant for deterministic ops
    node->eval(generator);
  }
  double log_prob = 0.0;
  for (auto node_id : sto_nodes) {
    Node* node = nodes[node_id].get();
    log_prob += node->log_prob();
  }
  return log_prob;
}

double Graph::full_log_prob() {
  _ensure_evaluation_and_inference_readiness();
  double sum_log_prob = 0.0;
  mt19937 generator(12131); // seed is irrelevant for deterministic ops
  for (auto node : mutable_support_ptrs()) {
    if (node->is_stochastic()) {
      sum_log_prob += node->log_prob();
      if (node->node_type == NodeType::OPERATOR) {
        auto sto_node = static_cast<oper::StochasticOperator*>(node);
        if (sto_node->transform_type != TransformType::NONE) {
          // If y = g(x), then by Change of Variables as using in statistics
          // (see references below),
          // then the density f_Y of Y can be computed from
          // the density f_X of X as
          // f_Y(y) = f_X(g^{-1}(y)) * |d/dy g^{-1}(y)|
          // log(f_Y(y)) = log(f_X(x)) + log(|d/dy f^{-1}(y)|)
          //   = node->log_prob() + log_abs_jacobian_determinant()
          // TODO: rename log_abs_jacobian_determinant
          // to log_abs_jacobian_determinant_of_inverse_transform
          //
          // References on Change of Variables in statistics:
          // https://online.stat.psu.edu/stat414/lesson/22/22.2
          // Stan reference:
          // https://mc-stan.org/docs/2_27/reference-manual/change-of-variables-section.html
          sum_log_prob += sto_node->log_abs_jacobian_determinant();
        }
      }
    } else {
      node->eval(generator);
    }
  }
  return sum_log_prob;
}

// TODO: from now on, we have methods for adding nodes, checking validity,
// inference and a copy constructor Those are essentially as they should be.
// Note that methods for determining support and affected nodes are in
// support.cpp, for not a very clear reason.

vector<Node*> Graph::convert_parent_ids(
    // TODO: this does not have to apply to parents only; make it a more general
    // function from ids to nodes.
    const vector<NodeID>& parent_ids) const {
  // check that the parent ids are valid indices and convert them to
  // an array of Node* pointers
  vector<Node*> parent_nodes;
  for (NodeID parent_id : parent_ids) {
    if (parent_id >= static_cast<NodeID>(nodes.size())) {
      throw out_of_range(
          "parent node_id " + std::to_string(parent_id) + "must be less than " +
          std::to_string(nodes.size()));
    }
    parent_nodes.push_back(nodes[parent_id].get());
  }
  return parent_nodes;
}

NodeID Graph::add_node(unique_ptr<Node> node) {
  NodeID index = static_cast<NodeID>(nodes.size());
  node->index = index;
  for (auto in_node : node->in_nodes) {
    in_node->out_nodes.push_back(node.get());
  }
  nodes.push_back(move(node));
  return index;
}

function<NodeID(NodeID)> Graph::remove_node(NodeID node_id) {
  return remove_node(nodes[node_id]);
}

function<NodeID(NodeID)> Graph::remove_node(unique_ptr<Node>& node) {
  if (!node->out_nodes.empty()) {
    throw invalid_argument(
        "Attempt to remove node with out-nodes. Node id = " +
        std::to_string(node->index));
  }

  auto equal_to_node = [&](Node* node2) { return node2->index == node->index; };
  for (auto& other_node : nodes) {
    if (other_node != node) {
      erase_if(other_node->out_nodes, equal_to_node);
    }
  }

  // Record node id because it will be destructed when we remove it from `nodes`
  // (since `nodes` is a vector of `unique_ptr`).
  auto node_id = node->index;
  auto max_id = nodes.size() - 1;

  // Remove it from everywhere
  erase(queries, node_id);
  observed.erase(node_id);
  erase_position(nodes, node_id);

  // Node ids no longer coincide with their positions, fix that.
  reindex_nodes();

  // auxiliary inference caches are invalidated
  ready_for_evaluation_and_inference = false;
  // Stored old values no longer valid
  _old_values_vector_has_the_right_size = false;

  // Map from old to new ids reflects that fact that
  // nodes with greater ids were shifted down one position:
  auto from_old_to_new_id = [removed_index{node_id}, max_id](NodeID id) {
    if (id == removed_index) {
      throw invalid_argument(
          "Looking up new id for old id after removed graph node "
          "but given id is the one for the removed node");
    }
    if (id > max_id) {
      throw invalid_argument(
          "Looking up new id for old id that is actually greater than maximum old id.");
    }
    return id > removed_index ? id - 1 : id;
  };

  return from_old_to_new_id;
}

void Graph::check_node_id(NodeID node_id) {
  if (node_id >= static_cast<NodeID>(nodes.size())) {
    throw out_of_range(
        "node_id (" + std::to_string(node_id) + ") must be less than " +
        std::to_string(nodes.size()));
  }
}

Node* Graph::get_node(NodeID node_id) {
  check_node_id(node_id);
  return nodes[node_id].get();
}

Node* Graph::check_node(NodeID node_id, NodeType node_type) {
  Node* node = get_node(node_id);
  if (node->node_type != node_type) {
    throw invalid_argument(
        "node_id " + std::to_string(node_id) + "expected type " +
        std::to_string(static_cast<int>(node_type)) + " but actual type " +
        std::to_string(static_cast<int>(node->node_type)));
  }
  return node;
}

Node* Graph::check_observed_node(NodeID node_id, bool is_scalar) {
  Node* node = get_node(node_id);
  if (node->node_type != NodeType::OPERATOR) {
    throw invalid_argument("only SAMPLE and IID_SAMPLE nodes may be observed");
  }
  oper::Operator* op = static_cast<oper::Operator*>(node);
  if (op->op_type != OperatorType::SAMPLE and
      op->op_type != OperatorType::IID_SAMPLE) {
    throw invalid_argument("only SAMPLE and IID_SAMPLE nodes may be observed");
  }
  if (observed.find(node_id) != observed.end()) {
    throw invalid_argument(
        "duplicate observe for node_id " + std::to_string(node_id));
  }

  if (is_scalar && node->value.type.variable_type != VariableType::SCALAR) {
    throw invalid_argument(
        "a matrix-valued sample may not be observed with a single value");
  }

  if (!is_scalar &&
      node->value.type.variable_type != VariableType::BROADCAST_MATRIX &&
      node->value.type.variable_type != VariableType::COL_SIMPLEX_MATRIX) {
    throw invalid_argument(
        "a scalar-valued sample may not be observed with a matrix value");
  }

  return node;
}

NodeID Graph::add_constant(bool value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant_bool(bool value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant(double value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant_real(double value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant(natural_t value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant_natural(natural_t value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant(NodeValue value) {
  unique_ptr<ConstNode> node = make_unique<ConstNode>(value);
  return add_node(move(node));
}

NodeID Graph::add_constant_probability(double value) {
  if (value < 0 or value > 1) {
    throw invalid_argument("probability must be between 0 and 1");
  }
  return add_constant(NodeValue(AtomicType::PROBABILITY, value));
}

NodeID Graph::add_constant_pos_real(double value) {
  if (value < 0) {
    throw invalid_argument("pos_real must be >=0");
  }
  return add_constant(NodeValue(AtomicType::POS_REAL, value));
}

NodeID Graph::add_constant_neg_real(double value) {
  if (value > 0) {
    throw invalid_argument("neg_real must be <=0");
  }
  return add_constant(NodeValue(AtomicType::NEG_REAL, value));
}

NodeID Graph::add_constant_bool_matrix(Eigen::MatrixXb& value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant_real_matrix(Eigen::MatrixXd& value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant_natural_matrix(Eigen::MatrixXn& value) {
  return add_constant(NodeValue(value));
}

NodeID Graph::add_constant_pos_matrix(Eigen::MatrixXd& value) {
  if ((value.array() < 0).any()) {
    throw invalid_argument("All elements in pos_matrix must be >=0");
  }
  return add_constant(NodeValue(AtomicType::POS_REAL, value));
}

NodeID Graph::add_constant_neg_matrix(Eigen::MatrixXd& value) {
  if ((value.array() > 0).any()) {
    throw invalid_argument("All elements in neg_matrix must be <=0");
  }
  return add_constant(NodeValue(AtomicType::NEG_REAL, value));
}

NodeID Graph::add_constant_col_simplex_matrix(Eigen::MatrixXd& value) {
  if ((value.array() < 0).any()) {
    throw invalid_argument("All elements in col_simplex_matrix must be >=0");
  }
  bool invalid_colsum =
      ((value.colwise().sum().array() - 1.0).abs() > PRECISION * value.rows())
          .any();
  if (invalid_colsum) {
    throw invalid_argument("All cols in col_simplex_matrix must sum to 1");
  }
  return add_constant(NodeValue(
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX,
          AtomicType::PROBABILITY,
          static_cast<uint>(value.rows()),
          static_cast<uint>(value.cols())),
      value));
}

NodeID Graph::add_constant_probability_matrix(Eigen::MatrixXd& value) {
  if ((value.array() < 0).any() or (value.array() > 1).any()) {
    throw invalid_argument(
        "All elements in probability_matrix must be between 0 and 1");
  }
  return add_constant(NodeValue(AtomicType::PROBABILITY, value));
}

NodeID Graph::add_distribution(
    DistributionType dist_type,
    AtomicType sample_type,
    vector<NodeID> parent_ids) {
  vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  // create a distribution node
  unique_ptr<Node> node = distribution::Distribution::new_distribution(
      dist_type, ValueType(sample_type), parent_nodes);
  // and add the node to the graph
  return add_node(move(node));
}

NodeID Graph::add_distribution(
    DistributionType dist_type,
    ValueType sample_type,
    vector<NodeID> parent_ids) {
  vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  // create a distribution node
  unique_ptr<Node> node = distribution::Distribution::new_distribution(
      dist_type, sample_type, parent_nodes);
  // and add the node to the graph
  return add_node(move(node));
}

NodeID Graph::add_operator(OperatorType op_type, vector<NodeID> parent_ids) {
  vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  unique_ptr<Node> node =
      oper::OperatorFactory::create_op(op_type, parent_nodes);
  return add_node(move(node));
}

NodeID Graph::add_factor(FactorType fac_type, vector<NodeID> parent_ids) {
  vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  unique_ptr<Node> node = factor::Factor::new_factor(fac_type, parent_nodes);
  NodeID node_id = add_node(move(node));
  // factors are both stochastic nodes and observed nodes
  Node* node2 = check_node(node_id, NodeType::FACTOR);
  node2->is_observed = true;
  observed.insert(node_id);
  return node_id;
}

void Graph::observe(NodeID node_id, bool value) {
  // A bool can only be a bool NodeValue, so we can just pass it along.
  observe(node_id, NodeValue(value));
}

void Graph::observe(NodeID node_id, double value) {
  Node* node = check_observed_node(node_id, true);
  switch (node->value.type.atomic_type) {
    case AtomicType::REAL:
      // The double is automatically in range
      break;
    case AtomicType::PROBABILITY:
    case AtomicType::POS_REAL:
    case AtomicType::NEG_REAL:
      // TODO: Add checks that the observed value is in range.
      break;
    default:
      throw invalid_argument(
          "observe expected " + node->value.type.to_string());
  }
  add_observe(node, NodeValue(node->value.type.atomic_type, value));
}

void Graph::observe(NodeID node_id, natural_t value) {
  // A natural can only be a natural NodeValue, so we can just pass it along.
  observe(node_id, NodeValue(value));
}

void Graph::observe(NodeID node_id, Eigen::MatrixXd& value) {
  Node* node = check_observed_node(node_id, false);
  // We know that we have a matrix value; is it the right shape?
  if (value.rows() != node->value.type.rows or
      value.cols() != node->value.type.cols) {
    throw invalid_argument("observe expected " + node->value.type.to_string());
  }
  switch (node->value.type.atomic_type) {
    case AtomicType::REAL:
      // The double is automatically in range
      break;
    case AtomicType::PROBABILITY:
    case AtomicType::POS_REAL:
    case AtomicType::NEG_REAL:
      // TODO: Add checks that the observed values are in range.
      // TODO: Check that an observed simplex is given a simplex.
      break;
    default:
      throw invalid_argument(
          "observe expected " + node->value.type.to_string());
  }

  add_observe(node, NodeValue(node->value.type, value));
}

void Graph::observe(NodeID node_id, Eigen::MatrixXb& value) {
  Node* node = check_observed_node(node_id, false);
  if (value.rows() != node->value.type.rows or
      value.cols() != node->value.type.cols or
      node->value.type.atomic_type != AtomicType::BOOLEAN) {
    throw invalid_argument(
        "observe expected a " + node->value.type.to_string());
  }
  add_observe(node, NodeValue(node->value.type, value));
}

void Graph::observe(NodeID node_id, Eigen::MatrixXn& value) {
  Node* node = check_observed_node(node_id, false);
  // We know that we have a matrix value; is it the right shape?
  if (value.rows() != node->value.type.rows or
      value.cols() != node->value.type.cols or
      node->value.type.atomic_type != AtomicType::NATURAL) {
    throw invalid_argument(
        "observe expected a " + node->value.type.to_string());
  }
  add_observe(node, NodeValue(node->value.type, value));
}

void Graph::observe(NodeID node_id, NodeValue value) {
  Node* node = check_observed_node(
      node_id, value.type.variable_type == VariableType::SCALAR);
  if (node->value.type != value.type) {
    throw invalid_argument(
        "observe expected " + node->value.type.to_string() + " but got " +
        value.type.to_string());
  }
  add_observe(node, value);
}

void Graph::add_observe(Node* node, NodeValue value) {
  // Precondition: node_id and value have already been checked
  // for validity.
  node->value = value;
  node->is_observed = true;
  observed.insert(node->index);
}

void Graph::customize_transformation(
    TransformType customized_type,
    vector<NodeID> node_ids) {
  if (common_transformations.empty()) {
    common_transformations[TransformType::LOG] = make_unique<transform::Log>();
    common_transformations[TransformType::SIGMOID] =
        make_unique<transform::Sigmoid>();
  }
  auto iter = common_transformations.find(customized_type);
  if (iter == common_transformations.end()) {
    throw invalid_argument("Unsupported transformation type.");
  }
  Transformation* transform_ptr = common_transformations[customized_type].get();
  for (auto node_id : node_ids) {
    auto node = check_node(node_id, NodeType::OPERATOR);
    if (not node->is_stochastic()) {
      throw invalid_argument(
          "Transformation only applies to Stochastic Operators.");
    }
    auto sto_node = static_cast<oper::StochasticOperator*>(node);

    switch (customized_type) {
      case TransformType::LOG:
        if (sto_node->value.type.atomic_type != AtomicType::POS_REAL) {
          throw invalid_argument("Log transformation requires POS_REAL value.");
        }
        break;
      case TransformType::SIGMOID:
        if (sto_node->value.type.atomic_type != AtomicType::PROBABILITY) {
          throw invalid_argument(
              "Sigmoid transformation requires PROBABILITY value.");
        }
        break;
      default:
        throw invalid_argument("Unsupported transformation type.");
    }
    sto_node->transform = transform_ptr;
    sto_node->transform_type = customized_type;
  }
}

void Graph::remove_observations() {
  // note that Factor nodes although technically observations are not
  // user-created observations and so these are not removed by this API
  for (auto itr = observed.begin(); itr != observed.end();) {
    Node* node = nodes[*itr].get();
    if (node->node_type != NodeType::FACTOR) {
      node->is_observed = false;
      itr = observed.erase(itr);
    } else {
      itr++;
    }
  }
}

NodeID Graph::query(NodeID node_id) {
  Node* node = get_node(node_id);
  NodeType t = node->node_type;
  if (t != NodeType::CONSTANT and t != NodeType::OPERATOR) {
    throw invalid_argument(
        "Query of node_id " + std::to_string(node_id) +
        " expected a node of type " +
        std::to_string(static_cast<int>(NodeType::CONSTANT)) + " or " +
        std::to_string(static_cast<int>(NodeType::OPERATOR)) + " but is " +
        std::to_string(static_cast<int>(t)));
  }
  // Adding a query is idempotent; querying the same node twice returns
  // the same query identifier both times.
  //
  // This is a linear search but the vector of queries is almost always
  // very short.
  auto it = find(queries.begin(), queries.end(), node_id);
  if (it != queries.end()) {
    return static_cast<NodeID>(it - queries.begin());
  }
  queries.push_back(node_id);
  return static_cast<NodeID>(queries.size() - 1); // the index is 0-based
}

void Graph::collect_log_prob(double log_prob) {
  auto& logprob_collector = (master_graph == nullptr)
      ? this->log_prob_vals
      : master_graph->log_prob_allchains[thread_index];
  logprob_collector.push_back(log_prob);
}

vector<vector<double>>& Graph::get_log_prob() {
  // TODO: clarify the meaning of log_prob_vals and log_prob_allchains
  // so we can check correctness of this method

  if (log_prob_vals.size() > 0) {
    log_prob_allchains.clear();
    log_prob_allchains.push_back(log_prob_vals);
  }
  return log_prob_allchains;
}

void Graph::collect_sample() {
  if (agg_type == AggregationType::NONE) {
    // construct a sample of the queried nodes
    auto& sample_collector = (master_graph == nullptr)
        ? this->samples
        : master_graph->samples_allchains[thread_index];
    vector<NodeValue> sample;
    for (NodeID node_id : queries) {
      sample.push_back(nodes[node_id]->value);
    }
    sample_collector.push_back(sample);
  }
  // note: we divide each new value by agg_samples rather than directly add
  // them to the total to avoid overflow
  else if (agg_type == AggregationType::MEAN) {
    auto& mean_collector = (master_graph == nullptr)
        ? this->means
        : master_graph->means_allchains[thread_index];
    assert(mean_collector.size() == queries.size());
    NodeID pos = 0;
    for (NodeID node_id : queries) {
      NodeValue value = nodes[node_id]->value;
      if (value.type == AtomicType::BOOLEAN) {
        mean_collector[pos] += double(value._bool) / agg_samples;
      } else if (
          value.type == AtomicType::REAL or
          value.type == AtomicType::POS_REAL or
          value.type == AtomicType::NEG_REAL or
          value.type == AtomicType::PROBABILITY) {
        mean_collector[pos] += value._double / agg_samples;
      } else if (value.type == AtomicType::NATURAL) {
        mean_collector[pos] += double(value._natural) / agg_samples;
      } else {
        throw runtime_error(
            "Mean aggregation only supported for "
            "boolean/real/probability/natural-valued nodes");
      }
      pos++;
    }
  } else {
    assert(false);
  }
}

void Graph::_infer(
    uint num_samples,
    InferenceType algorithm,
    uint seed,
    InferConfig infer_config) {
  if (queries.size() == 0) {
    throw runtime_error("no nodes queried for inference");
  }
  if (num_samples < 1) {
    throw runtime_error("num_samples can't be zero");
  }
  if (algorithm == InferenceType::REJECTION) {
    rejection(num_samples, seed, infer_config);
  } else if (algorithm == InferenceType::GIBBS) {
    gibbs(num_samples, seed, infer_config);
  } else if (algorithm == InferenceType::NMC) {
    nmc(num_samples, seed, infer_config);
  } else if (algorithm == InferenceType::NUTS) {
    nuts(num_samples, seed, infer_config);
  } else {
    throw invalid_argument("unsupported inference algorithm.");
  }
}

vector<vector<NodeValue>>&
Graph::infer(uint num_samples, InferenceType algorithm, uint seed) {
  InferConfig infer_config = InferConfig();
  // TODO: why don't the initialization below to be done for _infer?
  // If they do, move them there.
  // Not clear why we have infer and _infer instead of just infer with
  // a default infer_config.
  agg_type = AggregationType::NONE;
  samples.clear();
  log_prob_vals.clear();
  log_prob_allchains.clear();
  _infer(num_samples, algorithm, seed, infer_config);
  _produce_performance_report(num_samples, algorithm, seed);
  return samples;
}

vector<vector<vector<NodeValue>>>& Graph::infer(
    uint num_samples,
    InferenceType algorithm,
    uint seed,
    uint n_chains,
    InferConfig infer_config) {
  agg_type = AggregationType::NONE;
  samples.clear();
  samples_allchains.clear();
  samples_allchains.resize(n_chains, vector<vector<NodeValue>>());
  log_prob_vals.clear();
  log_prob_allchains.clear();
  log_prob_allchains.resize(n_chains, vector<double>());
  _infer_parallel(num_samples, algorithm, seed, n_chains, infer_config);
  _produce_performance_report(num_samples, algorithm, seed);
  return samples_allchains;
}

void Graph::_infer_parallel(
    uint num_samples,
    InferenceType algorithm,
    uint seed,
    uint n_chains,
    InferConfig infer_config) {
  if (n_chains < 1) {
    throw runtime_error("n_chains can't be zero");
  }
  master_graph = this;
  thread_index = 0;
  // clone graphs
  vector<Graph*> graph_copies;
  vector<uint> seedvec;
  for (uint i = 0; i < n_chains; i++) {
    if (i > 0) {
      Graph* g_ptr = new Graph(*this);
      g_ptr->thread_index = i;
      graph_copies.push_back(g_ptr);
    } else {
      graph_copies.push_back(this);
    }
    seedvec.push_back(seed + 13 * static_cast<uint>(i));
  }
  assert(graph_copies.size() == n_chains);
  assert(seedvec.size() == n_chains);
  // start threads
  vector<thread> threads;
  exception_ptr e = nullptr;
  for (uint i = 0; i < n_chains; i++) {
    thread infer_thread([&e,
                         &graph_copies,
                         i,
                         num_samples,
                         algorithm,
                         &seedvec,
                         infer_config]() {
      try {
        graph_copies[i]->_infer(
            num_samples, algorithm, seedvec[i], infer_config);
      } catch (...) {
        e = current_exception();
      }
    });
    threads.push_back(move(infer_thread));
  }
  assert(threads.size() == n_chains);
  // join threads
  for (uint i = 0; i < n_chains; i++) {
    threads[i].join();
    if (i > 0) {
      delete graph_copies[i];
    }
  }
  graph_copies.clear();
  threads.clear();
  master_graph = nullptr;
  if (e != nullptr) {
    rethrow_exception(e);
  }
}

vector<double>&
Graph::infer_mean(uint num_samples, InferenceType algorithm, uint seed) {
  InferConfig infer_config = InferConfig();
  agg_type = AggregationType::MEAN;
  agg_samples = num_samples;
  means.clear();
  means.resize(queries.size(), 0.0);
  log_prob_vals.clear();
  log_prob_allchains.clear();
  _infer(num_samples, algorithm, seed, infer_config);
  return means;
}

vector<vector<double>>& Graph::infer_mean(
    uint num_samples,
    InferenceType algorithm,
    uint seed,
    uint n_chains,
    InferConfig infer_config) {
  agg_type = AggregationType::MEAN;
  agg_samples = num_samples;
  means.clear();
  means.resize(queries.size(), 0.0);
  means_allchains.clear();
  means_allchains.resize(n_chains, vector<double>(queries.size(), 0.0));
  log_prob_vals.clear();
  log_prob_allchains.clear();
  _infer_parallel(num_samples, algorithm, seed, n_chains, infer_config);
  return means_allchains;
}

vector<vector<double>>& Graph::variational(
    uint num_iters,
    uint steps_per_iter,
    uint seed,
    uint elbo_samples) {
  if (queries.size() == 0) {
    throw runtime_error("no nodes queried for inference");
  }
  for (NodeID node_id : queries) {
    Node* node = nodes[node_id].get();
    if (not node->is_stochastic()) {
      throw invalid_argument(
          "only sample nodes may be queried in "
          "variational inference");
    }
  }
  elbo_vals.clear();
  mt19937 generator(seed);
  cavi(num_iters, steps_per_iter, generator, elbo_samples);
  return variational_params; // TODO: this should have been defined as a
                             // field, but a value returned by cavi.
}

vector<NodeID> Graph::get_parent_ids(const vector<Node*>& parent_nodes) const {
  vector<NodeID> parent_ids;
  for (auto node : parent_nodes) {
    parent_ids.push_back(node->index);
  }
  return parent_ids;
}

void Graph::reindex_nodes() {
  NodeID index = 0;
  for (auto const& node : nodes) {
    if (node) {
      node->index = index;
      index++;
    }
  }
}

Graph::Graph(const Graph& other) {
  // This copy constructor does not copy the inference results (if available)
  // from the source graph.
  *this = other;
}

Graph& Graph::operator=(const Graph& other) {
  // This copy assignment operator does not copy the inference results
  // (if available) from the source graph.
  for (NodeID i = 0; i < static_cast<NodeID>(other.nodes.size()); i++) {
    Node* node = other.nodes[i].get();
    vector<NodeID> parent_ids = get_parent_ids(node->in_nodes);
    switch (node->node_type) {
      case NodeType::CONSTANT: {
        NodeValue value_copy = NodeValue(node->value);
        add_constant(value_copy);
        break;
      }
      case NodeType::DISTRIBUTION: {
        distribution::Distribution* dist =
            static_cast<distribution::Distribution*>(node);
        add_distribution(dist->dist_type, dist->sample_type, parent_ids);
        break;
      }
      case NodeType::OPERATOR: {
        add_operator(static_cast<oper::Operator*>(node)->op_type, parent_ids);
        if (node->is_observed) {
          observe(node->index, NodeValue(node->value));
        }
        break;
      }
      case NodeType::FACTOR: {
        add_factor(static_cast<factor::Factor*>(node)->fac_type, parent_ids);
        break;
      }
      default: {
        throw invalid_argument("Trying to copy a node of unknown type.");
      }
    }
  }
  for (NodeID node_id : other.queries) {
    query(node_id);
  }
  master_graph = other.master_graph;
  agg_type = other.agg_type;
  agg_samples = other.agg_samples;

  return *this;
}

void Graph::_compute_evaluation_and_inference_readiness_data() {
  pd_begin(ProfilerEvent::NMC_INFER_INITIALIZE);
  _clear_evaluation_and_inference_readiness_data();
  _collect_node_ptrs();
  _collect_support();
  _collect_affected_operator_nodes();
  pd_finish(ProfilerEvent::NMC_INFER_INITIALIZE);
}

void Graph::_clear_evaluation_and_inference_readiness_data() {
  _node_ptrs.clear();
  _mutable_support.clear();
  _mutable_support_ptrs.clear();
  _unobserved_mutable_support.clear();
  _unobserved_sto_mutable_support.clear();
  _sto_affected_nodes.clear();
  _det_affected_operator_nodes.clear();
  unobserved_mutable_support_index_by_node_id.clear();
  unobserved_sto_mutable_support_index_by_node_id.clear();
}

void Graph::_collect_node_ptrs() {
  for (NodeID node_id = 0; node_id < static_cast<NodeID>(nodes.size());
       node_id++) {
    _node_ptrs.push_back(nodes[node_id].get());
  }
}

void Graph::_collect_support() {
  _mutable_support_ptrs.reserve(nodes.size());
  _mutable_support = compute_mutable_support();
  for (NodeID node_id : _mutable_support) {
    _mutable_support_ptrs.push_back(_node_ptrs[node_id]);
  }

  unobserved_mutable_support_index_by_node_id = vector<size_t>(nodes.size(), 0);
  unobserved_sto_mutable_support_index_by_node_id =
      vector<size_t>(nodes.size(), 0);

  for (auto node : _mutable_support_ptrs) {
    NodeID node_id = node->index;
    bool node_is_not_observed = observed.find(node->index) == observed.end();
    if (node_is_not_observed) {
      // NOLINTNEXTLINE
      unobserved_mutable_support_index_by_node_id[node_id] =
          _unobserved_mutable_support.size();
      _unobserved_mutable_support.push_back(node);
      if (node->is_stochastic()) {
        // NOLINTNEXTLINE
        unobserved_sto_mutable_support_index_by_node_id[node_id] =
            _unobserved_sto_mutable_support.size();
        _unobserved_sto_mutable_support.push_back(node);
      }
    }
  }
}

// For every unobserved stochastic node in the graph, we will need to
// repeatedly know the set of immediate stochastic descendants
// and intervening deterministic nodes.
// Because this can be expensive, we compute those sets once and cache them.
void Graph::_collect_affected_operator_nodes() {
  for (Node* node : _unobserved_sto_mutable_support) {
    auto det_node_ids = util::make_reserved_vector<NodeID>(nodes.size());
    auto sto_node_ids = util::make_reserved_vector<NodeID>(nodes.size());
    auto det_nodes = util::make_reserved_vector<Node*>(nodes.size());
    auto sto_nodes = util::make_reserved_vector<Node*>(nodes.size());

    tie(det_node_ids, sto_node_ids) =
        compute_affected_nodes(node->index, _mutable_support);
    for (NodeID id : det_node_ids) {
      det_nodes.push_back(_node_ptrs[id]);
    }
    for (NodeID id : sto_node_ids) {
      sto_nodes.push_back(_node_ptrs[id]);
    }
    _det_affected_operator_nodes.push_back(det_nodes);
    _sto_affected_nodes.push_back(sto_nodes);
    if (_collect_performance_data) {
      profiler_data.det_supp_count[node->index] =
          static_cast<int>(det_nodes.size());
    }
  }
}

const vector<Node*>& Graph::get_det_affected_operator_nodes(NodeID node_id) {
  return det_affected_operator_nodes()
      [unobserved_sto_mutable_support_index_by_node_id[node_id]];
}

const vector<Node*>& Graph::get_sto_affected_nodes(NodeID node_id) {
  return sto_affected_nodes()
      [unobserved_sto_mutable_support_index_by_node_id[node_id]];
}

void Graph::revertibly_set_and_propagate(Node* node, const NodeValue& value) {
  save_old_value(node);
  save_old_values(get_det_affected_operator_nodes(node));
  _old_sto_affected_nodes_log_prob =
      compute_log_prob_of(get_sto_affected_nodes(node));
  node->value = value;
  eval(get_det_affected_operator_nodes(node));
}

void Graph::revert_set_and_propagate(Node* node) {
  restore_old_value(node);
  restore_old_values(get_det_affected_operator_nodes(node));
}

void Graph::save_old_value(const Node* node) {
  _ensure_old_values_has_the_right_size();
  _old_values[node->index] = node->value;
}

void Graph::save_old_values(const vector<Node*>& nodes) {
  pd_begin(ProfilerEvent::NMC_SAVE_OLD);
  _ensure_old_values_has_the_right_size();
  for (Node* node : nodes) {
    _old_values[node->index] = node->value;
  }
  pd_finish(ProfilerEvent::NMC_SAVE_OLD);
}

NodeValue& Graph::get_old_value(const Node* node) {
  _check_old_values_are_valid();
  return _old_values[node->index];
}

void Graph::restore_old_value(Node* node) {
  _check_old_values_are_valid();
  node->value = _old_values[node->index];
}

void Graph::restore_old_values(const vector<Node*>& det_nodes) {
  pd_begin(ProfilerEvent::NMC_RESTORE_OLD);
  _check_old_values_are_valid();
  for (Node* node : det_nodes) {
    node->value = _old_values[node->index];
  }
  pd_finish(ProfilerEvent::NMC_RESTORE_OLD);
}

void Graph::compute_gradients(const vector<Node*>& det_nodes) {
  pd_begin(ProfilerEvent::NMC_COMPUTE_GRADS);
  for (Node* node : det_nodes) {
    node->compute_gradients();
  }
  pd_finish(ProfilerEvent::NMC_COMPUTE_GRADS);
}

void Graph::eval(const vector<Node*>& det_nodes) {
  pd_begin(ProfilerEvent::NMC_EVAL);
  mt19937 gen(12131); // seed doesn't matter
  // because operators are deterministic - TODO: clean it
  for (Node* node : det_nodes) {
    node->eval(gen);
  }
  pd_finish(ProfilerEvent::NMC_EVAL);
}

void Graph::clear_gradients(Node* node) {
  // TODO: eventually we want to have different classes of Node
  // and have this be a virtual method
  switch (node->value.type.variable_type) {
    case VariableType::SCALAR:
      node->grad1 = 0;
      node->grad2 = 0;
      break;
    case VariableType::BROADCAST_MATRIX:
    case VariableType::COL_SIMPLEX_MATRIX: {
      auto rows = node->value._matrix.rows();
      auto cols = node->value._matrix.cols();
      node->Grad1 = Eigen::MatrixXd::Zero(rows, cols);
      node->Grad2 = Eigen::MatrixXd::Zero(rows, cols);
      break;
    }
    default:
      throw runtime_error(
          "clear_gradients invoked for nodes of an unsupported variable type " +
          std::to_string(int(node->value.type.variable_type)));
  }
}

void Graph::clear_gradients(const vector<Node*>& nodes) {
  pd_begin(ProfilerEvent::NMC_CLEAR_GRADS);
  for (Node* node : nodes) {
    clear_gradients(node);
  }
  pd_finish(ProfilerEvent::NMC_CLEAR_GRADS);
}

void Graph::clear_gradients_of_node_and_its_affected_nodes(Node* node) {
  clear_gradients(node);
  clear_gradients(get_det_affected_operator_nodes(node));
  clear_gradients(get_sto_affected_nodes(node));
}

// Computes the log probability with respect to a given
// set of stochastic nodes.
double Graph::compute_log_prob_of(const vector<Node*>& sto_nodes) {
  double log_prob = 0;
  for (Node* node : sto_nodes) {
    log_prob += node->log_prob();
  }
  return log_prob;
}

bool are_equal(const Node& node1, const Node& node2) {
  using namespace distribution;
  using namespace oper;
  using namespace factor;

  auto const1 = dynamic_cast<const ConstNode*>(&node1);
  if (const1 != nullptr) {
    auto const2 = dynamic_cast<const ConstNode*>(&node2);
    return const2 != nullptr and const1->value == const2->value;
  }

  if (auto dist1 = dynamic_cast<const Distribution*>(&node1)) {
    auto dist2 = dynamic_cast<const Distribution*>(&node2);
    return dist2 != nullptr and dist1->dist_type == dist2->dist_type and
        node1.in_nodes == node2.in_nodes;
  }

  if (auto op1 = dynamic_cast<const Operator*>(&node1)) {
    auto op2 = dynamic_cast<const Operator*>(&node2);
    return op2 != nullptr and op1->op_type == op2->op_type and
        node1.in_nodes == node2.in_nodes;
  }

  if (auto fac1 = dynamic_cast<const Factor*>(&node1)) {
    auto fac2 = dynamic_cast<const Factor*>(&node2);
    return fac2 != nullptr and fac1->fac_type == fac2->fac_type and
        node1.in_nodes == node2.in_nodes;
  }

  throw invalid_argument(
      "are_equal(const Node& node1, const Node& node2): node1 is not instance of any supported subclass");
}

} // namespace graph
} // namespace beanmachine
