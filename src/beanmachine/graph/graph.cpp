/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <random>
#include <sstream>
#include <thread>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/transform/transform.h"

namespace beanmachine {
namespace graph {

std::string ValueType::to_string() const {
  std::string vtype;
  std::string atype;
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
  if (type == AtomicType::POS_REAL) {
    if (_double < PRECISION) {
      _double = PRECISION;
    }
  } else if (type == AtomicType::NEG_REAL) {
    if (_double > -PRECISION) {
      _double = -PRECISION;
    }
  } else if (type == AtomicType::PROBABILITY) {
    if (_double < PRECISION) {
      _double = PRECISION;
    } else if (_double > (1 - PRECISION)) {
      _double = 1 - PRECISION;
    }
  } else {
    // this API is only meant for POS_REAL, NEG_REAL, REAL and PROBABILITY
    // values
    if (type != AtomicType::REAL) {
      throw std::invalid_argument(
          "expect probability, pos_real, neg_real or real type with floating point value");
    }
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
        throw std::invalid_argument("Unsupported types for BROADCAST_MATRIX.");
    }
  } else if (type.variable_type == VariableType::COL_SIMPLEX_MATRIX) {
    _matrix = Eigen::MatrixXd::Ones(type.rows, type.cols) / type.rows;
  } else if (type.variable_type == VariableType::SCALAR) {
    this->init_scalar(type.atomic_type);
  } else {
    throw std::invalid_argument("Unsupported variable type.");
  }
}

std::string NodeValue::to_string() const {
  std::ostringstream os;
  std::string type_str = type.to_string() + " ";
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
    back_grad1._double = 0;
  } else {
    back_grad1._matrix.setZero(value.type.rows, value.type.cols);
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
      throw std::runtime_error("unsupported AtomicType to cast to scalar");
  }
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

void Graph::update_backgrad(std::vector<Node*>& ordered_supp) {
  for (auto node : ordered_supp) {
    node->reset_backgrad();
  }
  for (auto it = ordered_supp.rbegin(); it != ordered_supp.rend(); ++it) {
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
    uint tgt_idx,
    uint src_idx,
    uint seed,
    NodeValue& value,
    double& grad1,
    double& grad2) {
  // TODO: used for testing only, should integrate it with
  // whatever code is actually being used for eval and grad.
  if (src_idx >= static_cast<uint>(nodes.size())) {
    throw std::out_of_range("src_idx " + std::to_string(src_idx));
  }
  if (tgt_idx >= static_cast<uint>(nodes.size()) or tgt_idx <= src_idx) {
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

void Graph::_test_backgrad(
    std::set<uint>& supp,
    std::vector<DoubleMatrix*>& grad1) {
  for (auto it = supp.begin(); it != supp.end(); ++it) {
    Node* node = nodes[*it].get();
    node->reset_backgrad();
  }
  grad1.clear();
  for (auto it = supp.rbegin(); it != supp.rend(); ++it) {
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
  std::reverse(grad1.begin(), grad1.end());
}

void Graph::test_grad(std::vector<DoubleMatrix*>& grad1) {
  std::set<uint> supp = compute_support();
  _test_backgrad(supp, grad1);
}

void Graph::eval_and_grad(std::vector<DoubleMatrix*>& grad1, uint seed) {
  std::mt19937 generator(seed);
  std::set<uint> supp = compute_support();
  for (auto it = supp.begin(); it != supp.end(); ++it) {
    Node* node = nodes[*it].get();
    if (!node->is_observed) {
      node->eval(generator);
    }
  }
  _test_backgrad(supp, grad1);
}

void set_value(Eigen::MatrixXd& variable, double value) {
  variable.setConstant(value);
}
void set_value(double& variable, double value) {
  variable = value;
}

template <class T>
void Graph::gradient_log_prob(uint src_idx, T& grad1, T& grad2) {
  // TODO: As of May 2021, this method is being used for testing only.
  // Refactor code so that we test the code actually being used for the
  // normal functionality of the class.
  // If that is not possible, this is an indication that this is code
  // is only useful for testing, and should therefore
  // be moved to the testing code.
  Node* src_node = check_node(src_idx, NodeType::OPERATOR);
  if (not src_node->is_stochastic()) {
    throw std::runtime_error(
        "gradient_log_prob only supported on stochastic nodes");
  }
  uint size = src_node->value.type.cols * src_node->value.type.rows;
  bool is_src_scalar = (size == 0);
  // start gradient
  if (!is_src_scalar) {
    src_node->Grad1 = Eigen::MatrixXd::Ones(size, 1);
    src_node->Grad2 = Eigen::MatrixXd::Zero(size, 1);
  }
  src_node->grad1 = 1;
  src_node->grad2 = 0;

  auto supp = compute_support();
  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) = compute_affected_nodes(src_idx, supp);
  if (!is_src_scalar and det_nodes.size() > 0) {
    throw std::runtime_error(
        "compute_gradients has not been implemented for vector source node");
    // TODO: remove this error message and leave it to compute_gradients
    // to issue an error if needed.
  }
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    // passing generator for signature,
    // but it is irrelevant for deterministic nodes.
    // TODO: can we make signature use a default generator then?
    std::mt19937 generator(12131);
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
  if (!is_src_scalar) {
    src_node->Grad1.setZero();
  }
  src_node->grad1 = 0;
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    if (!is_src_scalar) {
      node->Grad1.setZero(1, 1);
      node->Grad2.setZero(1, 1);
    } else {
      node->grad1 = node->grad2 = 0;
    }
  }
}

template void
Graph::gradient_log_prob<double>(uint src_idx, double& grad1, double& grad2);

double Graph::log_prob(uint src_idx) {
  // TODO: also used in tests only
  Node* src_node = check_node(src_idx, NodeType::OPERATOR);
  if (not src_node->is_stochastic()) {
    throw std::runtime_error("log_prob only supported on stochastic nodes");
  }
  auto supp = compute_support();
  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) = compute_affected_nodes(src_idx, supp);
  for (auto node_id : det_nodes) {
    Node* node = nodes[node_id].get();
    std::mt19937 generator(12131); // seed is irrelevant for deterministic ops
    node->eval(generator);
  }
  double log_prob = 0.0;
  for (auto node_id : sto_nodes) {
    Node* node = nodes[node_id].get();
    log_prob += node->log_prob();
  }
  return log_prob;
}

// TODO: this is the one actually used in code (as opposed to full_log_prob used
// in testing only, so why the _?)
double Graph::_full_log_prob(std::vector<Node*>& ordered_supp) {
  double sum_log_prob = 0.0;
  std::mt19937 generator(12131); // seed is irrelevant for deterministic ops
  for (auto node : ordered_supp) {
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
          // to log_abs_jacobian_detesrminant_of_inverse_transform
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

/* TODO: used in testing only; it looks like there has not been a need for it in
 * actual code so far; notheless we can leave it here as it is a natural
 * operation */
double Graph::full_log_prob() {
  std::set<uint> supp = compute_support();
  std::vector<Node*> ordered_supp;
  for (uint node_id : supp) {
    ordered_supp.push_back(nodes[node_id].get());
  }
  return _full_log_prob(ordered_supp);
}

// TODO: from now on, we have methods for adding nodes, checking validity,
// inference and a copy constructor Those are essentially as they should be.
// Note that methods for determining support and affected nodes are in
// support.cpp, for not a very clear reason.

std::vector<Node*> Graph::convert_parent_ids(
    // TODO: this does not have to apply to parents only; make it a more general
    // function from ids to nodes.
    const std::vector<uint>& parent_ids) const {
  // check that the parent ids are valid indices and convert them to
  // an array of Node* pointers
  std::vector<Node*> parent_nodes;
  for (uint parent_id : parent_ids) {
    if (parent_id >= static_cast<uint>(nodes.size())) {
      throw std::out_of_range(
          "parent node_id " + std::to_string(parent_id) + "must be less than " +
          std::to_string(nodes.size()));
    }
    parent_nodes.push_back(nodes[parent_id].get());
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
  uint index = node->index = static_cast<uint>(nodes.size());
  nodes.push_back(std::move(node));
  return index;
}

void Graph::check_node_id(uint node_id) {
  if (node_id >= static_cast<uint>(nodes.size())) {
    throw std::out_of_range(
        "node_id (" + std::to_string(node_id) + ") must be less than " +
        std::to_string(nodes.size()));
  }
}

Node* Graph::get_node(uint node_id) {
  check_node_id(node_id);
  return nodes[node_id].get();
}

Node* Graph::check_node(uint node_id, NodeType node_type) {
  Node* node = get_node(node_id);
  if (node->node_type != node_type) {
    throw std::invalid_argument(
        "node_id " + std::to_string(node_id) + "expected type " +
        std::to_string(static_cast<int>(node_type)) + " but actual type " +
        std::to_string(static_cast<int>(node->node_type)));
  }
  return node;
}

Node* Graph::check_observed_node(uint node_id, bool is_scalar) {
  Node* node = get_node(node_id);
  if (node->node_type != NodeType::OPERATOR) {
    throw std::invalid_argument(
        "only SAMPLE and IID_SAMPLE nodes may be observed");
  }
  oper::Operator* op = static_cast<oper::Operator*>(node);
  if (op->op_type != OperatorType::SAMPLE and
      op->op_type != OperatorType::IID_SAMPLE) {
    throw std::invalid_argument(
        "only SAMPLE and IID_SAMPLE nodes may be observed");
  }
  if (observed.find(node_id) != observed.end()) {
    throw std::invalid_argument(
        "duplicate observe for node_id " + std::to_string(node_id));
  }

  if (is_scalar && node->value.type.variable_type != VariableType::SCALAR) {
    throw std::invalid_argument(
        "a matrix-valued sample may not be observed with a single value");
  }

  if (!is_scalar &&
      node->value.type.variable_type != VariableType::BROADCAST_MATRIX &&
      node->value.type.variable_type != VariableType::COL_SIMPLEX_MATRIX) {
    throw std::invalid_argument(
        "a scalar-valued sample may not be observed with a matrix value");
  }

  return node;
}

uint Graph::add_constant(bool value) {
  return add_constant(NodeValue(value));
}

uint Graph::add_constant(double value) {
  return add_constant(NodeValue(value));
}

uint Graph::add_constant(natural_t value) {
  return add_constant(NodeValue(value));
}

uint Graph::add_constant(NodeValue value) {
  std::unique_ptr<ConstNode> node = std::make_unique<ConstNode>(value);
  // constants don't have parents
  return add_node(std::move(node), std::vector<uint>());
}

uint Graph::add_constant_probability(double value) {
  if (value < 0 or value > 1) {
    throw std::invalid_argument("probability must be between 0 and 1");
  }
  return add_constant(NodeValue(AtomicType::PROBABILITY, value));
}

uint Graph::add_constant_pos_real(double value) {
  if (value < 0) {
    throw std::invalid_argument("pos_real must be >=0");
  }
  return add_constant(NodeValue(AtomicType::POS_REAL, value));
}

uint Graph::add_constant_neg_real(double value) {
  if (value > 0) {
    throw std::invalid_argument("neg_real must be <=0");
  }
  return add_constant(NodeValue(AtomicType::NEG_REAL, value));
}

uint Graph::add_constant_bool_matrix(Eigen::MatrixXb& value) {
  return add_constant(NodeValue(value));
}

uint Graph::add_constant_real_matrix(Eigen::MatrixXd& value) {
  return add_constant(NodeValue(value));
}

uint Graph::add_constant_natural_matrix(Eigen::MatrixXn& value) {
  return add_constant(NodeValue(value));
}

uint Graph::add_constant_pos_matrix(Eigen::MatrixXd& value) {
  if ((value.array() < 0).any()) {
    throw std::invalid_argument("All elements in pos_matrix must be >=0");
  }
  return add_constant(NodeValue(AtomicType::POS_REAL, value));
}

uint Graph::add_constant_neg_matrix(Eigen::MatrixXd& value) {
  if ((value.array() > 0).any()) {
    throw std::invalid_argument("All elements in neg_matrix must be <=0");
  }
  return add_constant(NodeValue(AtomicType::NEG_REAL, value));
}

uint Graph::add_constant_col_simplex_matrix(Eigen::MatrixXd& value) {
  if ((value.array() < 0).any()) {
    throw std::invalid_argument(
        "All elements in col_simplex_matrix must be >=0");
  }
  bool invalid_colsum =
      ((value.colwise().sum().array() - 1.0).abs() > PRECISION * value.rows())
          .any();
  if (invalid_colsum) {
    throw std::invalid_argument("All cols in col_simplex_matrix must sum to 1");
  }
  return add_constant(NodeValue(
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX,
          AtomicType::PROBABILITY,
          static_cast<uint>(value.rows()),
          static_cast<uint>(value.cols())),
      value));
}

uint Graph::add_constant_probability_matrix(Eigen::MatrixXd& value) {
  if ((value.array() < 0).any() or (value.array() > 1).any()) {
    throw std::invalid_argument(
        "All elements in probability_matrix must be between 0 and 1");
  }
  return add_constant(NodeValue(AtomicType::PROBABILITY, value));
}

uint Graph::add_distribution(
    DistributionType dist_type,
    AtomicType sample_type,
    std::vector<uint> parent_ids) {
  std::vector<Node*> parent_nodes = convert_parent_ids(parent_ids);
  // create a distribution node
  std::unique_ptr<Node> node = distribution::Distribution::new_distribution(
      dist_type, ValueType(sample_type), parent_nodes);
  // and add the node to the graph
  return add_node(std::move(node), parent_ids);
}

uint Graph::add_distribution(
    DistributionType dist_type,
    ValueType sample_type,
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
      oper::OperatorFactory::create_op(op_type, parent_nodes);
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

void Graph::observe(uint node_id, bool value) {
  // A bool can only be a bool NodeValue, so we can just pass it along.
  observe(node_id, NodeValue(value));
}

void Graph::observe(uint node_id, double value) {
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
      throw std::invalid_argument(
          "observe expected " + node->value.type.to_string());
  }
  add_observe(node, NodeValue(node->value.type.atomic_type, value));
}

void Graph::observe(uint node_id, natural_t value) {
  // A natural can only be a natural NodeValue, so we can just pass it along.
  observe(node_id, NodeValue(value));
}

void Graph::observe(uint node_id, Eigen::MatrixXd& value) {
  Node* node = check_observed_node(node_id, false);
  // We know that we have a matrix value; is it the right shape?
  if (value.rows() != node->value.type.rows or
      value.cols() != node->value.type.cols) {
    throw std::invalid_argument(
        "observe expected " + node->value.type.to_string());
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
      throw std::invalid_argument(
          "observe expected " + node->value.type.to_string());
  }

  add_observe(node, NodeValue(node->value.type, value));
}

void Graph::observe(uint node_id, Eigen::MatrixXb& value) {
  Node* node = check_observed_node(node_id, false);
  if (value.rows() != node->value.type.rows or
      value.cols() != node->value.type.cols or
      node->value.type.atomic_type != AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "observe expected a " + node->value.type.to_string());
  }
  add_observe(node, NodeValue(node->value.type, value));
}

void Graph::observe(uint node_id, Eigen::MatrixXn& value) {
  Node* node = check_observed_node(node_id, false);
  // We know that we have a matrix value; is it the right shape?
  if (value.rows() != node->value.type.rows or
      value.cols() != node->value.type.cols or
      node->value.type.atomic_type != AtomicType::NATURAL) {
    throw std::invalid_argument(
        "observe expected a " + node->value.type.to_string());
  }
  add_observe(node, NodeValue(node->value.type, value));
}

void Graph::observe(uint node_id, NodeValue value) {
  Node* node = check_observed_node(
      node_id, value.type.variable_type == VariableType::SCALAR);
  if (node->value.type != value.type) {
    throw std::invalid_argument(
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
    std::vector<uint> node_ids) {
  if (common_transformations.empty()) {
    common_transformations[TransformType::LOG] =
        std::make_unique<transform::Log>();
  }
  auto iter = common_transformations.find(customized_type);
  if (iter == common_transformations.end()) {
    throw std::invalid_argument("Unsupported transformation type.");
  }
  Transformation* transform_ptr = common_transformations[customized_type].get();
  for (auto node_id : node_ids) {
    auto node = check_node(node_id, NodeType::OPERATOR);
    if (not node->is_stochastic()) {
      throw std::invalid_argument(
          "Transformation only applies to Stochastic Operators.");
    }
    auto sto_node = static_cast<oper::StochasticOperator*>(node);

    switch (customized_type) {
      case TransformType::LOG:
        if (sto_node->value.type.atomic_type != AtomicType::POS_REAL) {
          throw std::invalid_argument(
              "Log transformation requires POS_REAL value.");
        }
        break;
      default:
        throw std::invalid_argument("Unsupported transformation type.");
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

uint Graph::query(uint node_id) {
  Node* node = get_node(node_id);
  NodeType t = node->node_type;
  if (t != NodeType::CONSTANT and t != NodeType::OPERATOR) {
    throw std::invalid_argument(
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
  auto it = std::find(queries.begin(), queries.end(), node_id);
  if (it != queries.end()) {
    return static_cast<uint>(it - queries.begin());
  }
  queries.push_back(node_id);
  return static_cast<uint>(queries.size() - 1); // the index is 0-based
}

void Graph::collect_log_prob(double log_prob) {
  auto& logprob_collector = (master_graph == nullptr)
      ? this->log_prob_vals
      : master_graph->log_prob_allchains[thread_index];
  logprob_collector.push_back(log_prob);
}

std::vector<std::vector<double>>& Graph::get_log_prob() {
  // TODO: clarify the meaning of log_prob_vals and log_prob_allchains
  // so we can check correctness of this method

  if (log_prob_vals.size() > 0) {
    log_prob_allchains.clear();
    log_prob_allchains.push_back(log_prob_vals);
  }
  return log_prob_allchains;
}

void Graph::collect_sample() {
  // construct a sample of the queried nodes
  auto& sample_collector = (master_graph == nullptr)
      ? this->samples
      : master_graph->samples_allchains[thread_index];
  auto& mean_collector = (master_graph == nullptr)
      ? this->means
      : master_graph->means_allchains[thread_index];
  if (agg_type == AggregationType::NONE) {
    std::vector<NodeValue> sample;
    for (uint node_id : queries) {
      sample.push_back(nodes[node_id]->value);
    }
    sample_collector.push_back(sample);
  }
  // note: we divide each new value by agg_samples rather than directly add
  // them to the total to avoid overflow
  else if (agg_type == AggregationType::MEAN) {
    assert(mean_collector.size() == queries.size());
    uint pos = 0;
    for (uint node_id : queries) {
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

void Graph::_infer(
    uint num_samples,
    InferenceType algorithm,
    uint seed,
    InferConfig infer_config) {
  if (queries.size() == 0) {
    throw std::runtime_error("no nodes queried for inference");
  }
  if (num_samples < 1) {
    throw std::runtime_error("num_samples can't be zero");
  }
  if (algorithm == InferenceType::REJECTION) {
    rejection(num_samples, seed, infer_config);
  } else if (algorithm == InferenceType::GIBBS) {
    gibbs(num_samples, seed, infer_config);
  } else if (algorithm == InferenceType::NMC) {
    nmc(num_samples, seed, infer_config);
  }
}

std::vector<std::vector<NodeValue>>&
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

std::vector<std::vector<std::vector<NodeValue>>>& Graph::infer(
    uint num_samples,
    InferenceType algorithm,
    uint seed,
    uint n_chains,
    InferConfig infer_config) {
  agg_type = AggregationType::NONE;
  samples.clear();
  samples_allchains.clear();
  samples_allchains.resize(n_chains, std::vector<std::vector<NodeValue>>());
  log_prob_vals.clear();
  log_prob_allchains.clear();
  log_prob_allchains.resize(n_chains, std::vector<double>());
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
    throw std::runtime_error("n_chains can't be zero");
  }
  master_graph = this;
  thread_index = 0;
  // clone graphs
  std::vector<Graph*> graph_copies;
  std::vector<uint> seedvec;
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
  std::vector<std::thread> threads;
  std::exception_ptr e = nullptr;
  for (uint i = 0; i < n_chains; i++) {
    std::thread infer_thread([&e,
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
        e = std::current_exception();
      }
    });
    threads.push_back(std::move(infer_thread));
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
    std::rethrow_exception(e);
  }
}

std::vector<double>&
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

std::vector<std::vector<double>>& Graph::infer_mean(
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
  means_allchains.resize(n_chains, std::vector<double>(queries.size(), 0.0));
  log_prob_vals.clear();
  log_prob_allchains.clear();
  _infer_parallel(num_samples, algorithm, seed, n_chains, infer_config);
  return means_allchains;
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
  return variational_params; // TODO: this should have been defined as a field,
                             // but a value returned by cavi.
}

std::vector<uint> Graph::get_parent_ids(
    const std::vector<Node*>& parent_nodes) const {
  std::vector<uint> parent_ids;
  for (auto node : parent_nodes) {
    parent_ids.push_back(node->index);
  }
  return parent_ids;
}

Graph::Graph(const Graph& other) {
  // This copy constructor does not copy the inference results (if available)
  // from the source graph.
  for (uint i = 0; i < static_cast<uint>(other.nodes.size()); i++) {
    Node* node = other.nodes[i].get();
    std::vector<uint> parent_ids = get_parent_ids(node->in_nodes);
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
        throw std::invalid_argument("Trying to copy a node of unknown type.");
      }
    }
  }
  for (uint node_id : other.queries) {
    query(node_id);
  }
  master_graph = other.master_graph;
  agg_type = other.agg_type;
  agg_samples = other.agg_samples;
}

} // namespace graph
} // namespace beanmachine
