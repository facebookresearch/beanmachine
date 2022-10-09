/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#define _USE_MATH_DEFINES
#include <cmath>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include "beanmachine/graph/double_matrix.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/third-party/nameof.h"
#include "beanmachine/graph/transformation.h"
#include "beanmachine/graph/util.h"

#define NATURAL_TYPE unsigned long long int
#ifdef _MSC_VER
#define uint unsigned int
#endif

namespace Eigen {
typedef Matrix<bool, Dynamic, Dynamic> MatrixXb;
typedef Matrix<NATURAL_TYPE, Dynamic, Dynamic> MatrixXn;
} // namespace Eigen

namespace beanmachine {
namespace graph {

const double PRECISION = 1e-10; // minimum precision of values

enum class VariableType {
  UNKNOWN, // For error catching
  SCALAR,
  BROADCAST_MATRIX,
  COL_SIMPLEX_MATRIX,
};

enum class AtomicType {
  UNKNOWN, // This is for error catching
  BOOLEAN,
  PROBABILITY,
  REAL,
  POS_REAL, // Real numbers greater than *or* equal to zero
  NATURAL, // note: NATURAL numbers include zero (ISO 80000-2)
  NEG_REAL, // Real numbers less than *or* equal to zero
};

struct ValueType {
  VariableType variable_type;
  AtomicType atomic_type;
  uint rows;
  uint cols;

  ValueType()
      : variable_type(VariableType::UNKNOWN),
        atomic_type(AtomicType::UNKNOWN),
        rows(0),
        cols(0) {}
  ValueType(const ValueType& other)
      : variable_type(other.variable_type),
        atomic_type(other.atomic_type),
        rows(other.rows),
        cols(other.cols) {}
  explicit ValueType(const AtomicType& other)
      : variable_type(VariableType::SCALAR),
        atomic_type(other),
        rows(0),
        cols(0) {}

  ValueType(VariableType vtype, AtomicType atype, uint rows, uint cols)
      : variable_type(vtype), atomic_type(atype), rows(rows), cols(cols) {
    if (vtype == VariableType::COL_SIMPLEX_MATRIX) {
      assert(atype == AtomicType::PROBABILITY);
    }
  }

  bool operator!=(const ValueType& other) const {
    if (variable_type != other.variable_type or
        atomic_type != other.atomic_type) {
      return true;
    } else if (
        variable_type == VariableType::SCALAR or
        variable_type == VariableType::UNKNOWN) {
      return false;
    } else {
      return rows != other.rows or cols != other.cols;
    }
  }
  bool operator!=(const AtomicType& other) const {
    return variable_type != VariableType::SCALAR or atomic_type != other;
  }
  bool operator==(const ValueType& other) const {
    return not(*this != other);
  }
  bool operator==(const AtomicType& other) const {
    return variable_type == VariableType::SCALAR and atomic_type == other;
  }
  ValueType& operator=(const ValueType& other) {
    if (this != &other) {
      variable_type = other.variable_type;
      atomic_type = other.atomic_type;
      rows = other.rows;
      cols = other.cols;
    }
    return *this;
  }
  ValueType& operator=(const AtomicType& other) {
    variable_type = VariableType::SCALAR;
    atomic_type = other;
    return *this;
  }
  std::string to_string() const;
};

inline bool atomic_type_unknown_or_equal_to(
    graph::AtomicType a,
    graph::ValueType v) {
  return a == graph::AtomicType::UNKNOWN or graph::ValueType(a) == v;
}

typedef NATURAL_TYPE natural_t;

extern NATURAL_TYPE NATURAL_ZERO;
extern NATURAL_TYPE NATURAL_ONE;

class NodeValue {
 public:
  ValueType type;
  union {
    bool _bool;
    double _double;
    natural_t _natural;
  };
  // In principle, the following fields should be in the above union.
  // However, because they have non-trivial destructors,
  // the programmer needs to explicitly define a destructor for the union
  // (one that knows which field is actually used so it can destruct only that).
  // However, anonymous unions cannot have member functions, including
  // destructors. One could make it a named union external to this class but
  // this would require redirecting all usage to the named union's fields,
  // which does not seem worth it.
  // See https://en.cppreference.com/w/cpp/language/union: "If a union
  // contains a non-static data member with a non-trivial special member
  // function (copy/move constructor, copy/move assignment, or destructor), that
  // function is deleted by default in the union and needs to be defined
  // explicitly by the programmer."
  Eigen::MatrixXd _matrix;
  Eigen::MatrixXb _bmatrix;
  Eigen::MatrixXn _nmatrix;

  NodeValue() : type(AtomicType::UNKNOWN) {}
  explicit NodeValue(AtomicType type);
  explicit NodeValue(ValueType type);
  explicit NodeValue(bool value) : type(AtomicType::BOOLEAN), _bool(value) {}
  explicit NodeValue(double value) : type(AtomicType::REAL), _double(value) {}
  explicit NodeValue(natural_t value)
      : type(AtomicType::NATURAL), _natural(value) {}
  explicit NodeValue(Eigen::MatrixXd& value)
      : type(ValueType(
            VariableType::BROADCAST_MATRIX,
            AtomicType::REAL,
            static_cast<int>(value.rows()),
            static_cast<int>(value.cols()))),
        _matrix(value) {}
  explicit NodeValue(Eigen::MatrixXb& value)
      : type(ValueType(
            VariableType::BROADCAST_MATRIX,
            AtomicType::BOOLEAN,
            static_cast<int>(value.rows()),
            static_cast<int>(value.cols()))),
        _bmatrix(value) {}
  explicit NodeValue(Eigen::MatrixXn& value)
      : type(ValueType(
            VariableType::BROADCAST_MATRIX,
            AtomicType::NATURAL,
            static_cast<int>(value.rows()),
            static_cast<int>(value.cols()))),
        _nmatrix(value) {}

  NodeValue(AtomicType type, bool value) : type(type), _bool(value) {
    assert(type == AtomicType::BOOLEAN);
  }
  NodeValue(AtomicType type, natural_t value) : type(type), _natural(value) {
    assert(type == AtomicType::NATURAL);
  }
  NodeValue(AtomicType type, Eigen::MatrixXd& value)
      : type(ValueType(
            VariableType::BROADCAST_MATRIX,
            type,
            static_cast<int>(value.rows()),
            static_cast<int>(value.cols()))),
        _matrix(value) {
    assert(
        type == AtomicType::REAL or type == AtomicType::POS_REAL or
        type == AtomicType::NEG_REAL or type == AtomicType::PROBABILITY);
  }
  NodeValue(AtomicType /* type */, Eigen::MatrixXb& value) : NodeValue(value) {}
  NodeValue(AtomicType /* type */, Eigen::MatrixXn& value) : NodeValue(value) {}
  NodeValue(ValueType type, Eigen::MatrixXd& value)
      : type(type), _matrix(value) {
    assert(
        type.variable_type == VariableType::BROADCAST_MATRIX or
        type.variable_type == VariableType::COL_SIMPLEX_MATRIX);
    assert(
        type.atomic_type == AtomicType::REAL or
        type.atomic_type == AtomicType::POS_REAL or
        type.atomic_type == AtomicType::NEG_REAL or
        type.atomic_type == AtomicType::PROBABILITY);
    assert(type.rows == value.rows() and type.cols == value.cols());
  }
  NodeValue(ValueType type, Eigen::MatrixXb& value)
      : type(type), _bmatrix(value) {
    assert(type.variable_type == VariableType::BROADCAST_MATRIX);
    assert(type.atomic_type == AtomicType::BOOLEAN);
    assert(type.rows == value.rows() and type.cols == value.cols());
  }
  NodeValue(ValueType type, Eigen::MatrixXn& value)
      : type(type), _nmatrix(value) {
    assert(type.variable_type == VariableType::BROADCAST_MATRIX);
    assert(type.atomic_type == AtomicType::NATURAL);
    assert(type.rows == value.rows() and type.cols == value.cols());
  }
  NodeValue(AtomicType type, double value);

  NodeValue(const NodeValue& other) : type(other.type) {
    switch (type.variable_type) {
      case VariableType::SCALAR:
        switch (type.atomic_type) {
          case AtomicType::UNKNOWN:
            // This used to throw an error but that
            // was unnecessarily restrictive.
            break;
          case AtomicType::BOOLEAN:
            _bool = other._bool;
            break;
          case AtomicType::NATURAL:
            _natural = other._natural;
            break;
          default:
            _double = other._double;
            break;
        }
        break;
      case VariableType::BROADCAST_MATRIX:
        switch (type.atomic_type) {
          case AtomicType::BOOLEAN:
            _bmatrix = other._bmatrix;
            break;
          case AtomicType::REAL:
          case AtomicType::POS_REAL:
          case AtomicType::NEG_REAL:
          case AtomicType::PROBABILITY:
            _matrix = other._matrix;
            break;
          case AtomicType::NATURAL:
            _nmatrix = other._nmatrix;
            break;
          default:
            throw std::invalid_argument(
                "Trying to copy a MATRIX NodeValue of unsupported type.");
        }
        break;
      case VariableType::COL_SIMPLEX_MATRIX:
        _matrix = other._matrix;
        break;
      default:
        throw std::invalid_argument(
            "Trying to copy a value of unknown VariableType");
    }
  }

  NodeValue& operator=(const NodeValue& other) = default;

  std::string to_string() const;
  bool operator==(const NodeValue& other) const {
    return type == other.type and
        ((type == AtomicType::BOOLEAN and _bool == other._bool) or
         (type == AtomicType::REAL and _double == other._double) or
         (type == AtomicType::POS_REAL and _double == other._double) or
         (type == AtomicType::NEG_REAL and _double == other._double) or
         (type == AtomicType::PROBABILITY and _double == other._double) or
         (type == AtomicType::NATURAL and _natural == other._natural) or
         (type.variable_type == VariableType::BROADCAST_MATRIX and
          (type.atomic_type == AtomicType::REAL or
           type.atomic_type == AtomicType::POS_REAL or
           type.atomic_type == AtomicType::NEG_REAL or
           type.atomic_type == AtomicType::PROBABILITY) and
          _matrix.isApprox(other._matrix)) or
         (type.variable_type == VariableType::BROADCAST_MATRIX and
          type.atomic_type == AtomicType::BOOLEAN and
          _bmatrix == other._bmatrix) or
         (type.variable_type == VariableType::BROADCAST_MATRIX and
          type.atomic_type == AtomicType::NATURAL and
          _nmatrix == other._nmatrix) or
         (type.variable_type == VariableType::COL_SIMPLEX_MATRIX and
          _matrix.isApprox(other._matrix)));
  }
  bool operator!=(const NodeValue& other) const {
    return not(*this == other);
  }

 private:
  void init_scalar(AtomicType type);
};

enum class OperatorType {
  UNKNOWN,
  SAMPLE, // This is the ~ operator in models.
          // IMPORTANT: always update the
          // first non-UNKNOWN type in the iterator below.
  IID_SAMPLE,
  TO_REAL,
  TO_POS_REAL,
  COMPLEMENT,
  NEGATE,
  ELEMENTWISE_MULTIPLY,
  EXP,
  EXPM1,
  MULTIPLY,
  ADD,
  MATRIX_NEGATE,
  PHI,
  LOGISTIC,
  IF_THEN_ELSE,
  LOG1PEXP,
  LOGSUMEXP,
  LOGSUMEXP_VECTOR,
  LOG,
  POW,
  LOG1MEXP,
  TRANSPOSE,
  MATRIX_MULTIPLY,
  MATRIX_SCALE,
  MATRIX_ADD,
  TO_PROBABILITY,
  INDEX,
  COLUMN_INDEX,
  TO_MATRIX,
  BROADCAST_ADD,
  TO_REAL_MATRIX,
  TO_POS_REAL_MATRIX,
  TO_NEG_REAL,
  TO_NEG_REAL_MATRIX,
  CHOICE,
  TO_INT,
  CHOLESKY,
  MATRIX_EXP,
  LOG_PROB,
  MATRIX_SUM,
  MATRIX_LOG,
  LOG1P,
  MATRIX_LOG1P,
  MATRIX_LOG1MEXP,
  MATRIX_PHI,
  MATRIX_COMPLEMENT,
  BROADCAST,
  FILL_MATRIX,
  // IMPORTANT: always update the last type in the iterator below.
};

using OperatorTypeIterable = util::EnumClassIterable<
    OperatorType,
    OperatorType::SAMPLE,
    OperatorType::FILL_MATRIX>;

enum class DistributionType {
  UNKNOWN,
  TABULAR, // IMPORTANT: always update the first non-UNKNOWN type in the
           // iterator below.
  BERNOULLI,
  BERNOULLI_NOISY_OR,
  BETA,
  BINOMIAL,
  DIRICHLET,
  FLAT,
  NORMAL,
  LOG_NORMAL,
  HALF_NORMAL,
  MULTIVARIATE_NORMAL,
  HALF_CAUCHY,
  STUDENT_T,
  BERNOULLI_LOGIT,
  GAMMA,
  BIMIXTURE,
  CATEGORICAL,
  POISSON,
  GEOMETRIC,
  CAUCHY,
  DUMMY,
  PRODUCT,
  LKJ_CHOLESKY, // IMPORTANT: always update the last type in the iterator below.
};
using DistributionTypeIterable = util::EnumClassIterable<
    DistributionType,
    DistributionType::TABULAR,
    DistributionType::LKJ_CHOLESKY>;

// TODO: do we really need DistributionType? Can't we know the type of a
// Distribution from its class alone?

enum class FactorType {
  UNKNOWN,
  EXP_PRODUCT,
};
using FactorTypeIterable = util::EnumClassIterable<
    FactorType,
    FactorType::EXP_PRODUCT,
    FactorType::EXP_PRODUCT>;

enum class NodeType {
  UNKNOWN,
  CONSTANT,
  DISTRIBUTION,
  OPERATOR,
  FACTOR,
  MAX,
};

enum class InferenceType {
  UNKNOWN,
  REJECTION,
  GIBBS,
  NMC,
  NUTS,
};

enum class AggregationType {
  UNKNOWN,
  NONE,
  MEAN,
};

struct InferConfig {
  bool keep_log_prob;
  double path_length;
  double step_size;
  uint num_warmup;
  bool keep_warmup;

  ~InferConfig() {}
  InferConfig(
      bool keep_log_prob = false,
      double path_length = 1.0,
      double step_size = 1.0,
      uint num_warmup = 0,
      bool keep_warmup = false)
      : keep_log_prob(keep_log_prob),
        path_length(path_length),
        step_size(step_size),
        num_warmup(num_warmup),
        keep_warmup(keep_warmup) {}
};

using NodeID = uint;

class Node {
 public:
  /*** Structural properties ***/

  NodeType node_type;
  std::vector<Node*> in_nodes;
  std::vector<Node*> out_nodes;

  /*** Stateful properties (to be eventually moved out) ***/

  NodeID index; // index in Graph::nodes
  bool is_observed = false;
  NodeValue value;
  double grad1;
  double grad2;
  Eigen::MatrixXd Grad1;
  Eigen::MatrixXd Grad2;
  DoubleMatrix back_grad1;

  /*** Constructors and destructor ***/

  explicit Node(const std::vector<Node*>& in_nodes)
      : Node(NodeType(), NodeValue(), in_nodes) {}

  Node(NodeType node_type, const std::vector<Node*>& in_nodes)
      : Node(node_type, NodeValue(), in_nodes) {}

  Node(NodeType node_type, NodeValue value, const std::vector<Node*>& in_nodes);

  virtual ~Node() {}

  /*** Cloning ***/

  virtual std::unique_ptr<Node> clone() = 0;

  /*** To string ***/

  /* A conversion to string, mostly for debugging purposes. */
  virtual std::string to_string() = 0;

  /*** Evaluation and gradients ***/

  virtual bool is_stochastic() const = 0;

  /* A mutable node is a node whose value or log prob needs to be
     updated when a node affecting them changes its value.
     See Graph::compute_support Graph::compute_affected_nodes
     for more information.
  */
  virtual bool is_mutable() const = 0;

  /*
   Evaluate the node and store the result in `value` if appropriate
   eval may involve sampling and that's why we need the random number engine.
  */
  virtual void eval(std::mt19937& gen) = 0;

  virtual bool needs_gradient() const {
    return true;
  }

  /*
   Computes the first and second gradients of this node
   with respect to some (unspecified -- see below) variable.
   More specifically, it uses the values stored in this node's input
   nodes grad1 and grad2 fields (or Grad1 and Grad2 if they are matrices)
   to compute the value of this node's own gradient fields
   (again grad1, grad2 or Grad1 and Grad2).
   Note that this method does *not* compute the gradient of
   this node with respect to its inputs,
   but with respect to some (possibly distant) variable.
   The method is neutral regarding which variable this is,
   and simply computes its own gradient with respect
   to the same variable its input node gradients are
   with respect to.
   In the planned refactoring of BMG's autograd (as of May 2022)
   this should be replaced by the more fundamental and modular function
   computing a node's gradient with respect to its own inputs,
   which should then be used as needed in
   applications of the chain rule.
  */
  virtual void compute_gradients() {}

  /*
   Gradient backward propagation: computes the 1st-order gradient update and
   add it to the parent's back_grad1.
  */
  virtual void backward() {}
  void reset_backgrad();

  /*
   The generic forward gradient propagation thru a node (deterministic operator
   or distribution) with multiple scalar inputs and one scalar output, w.r.t a
   scalar source. Template is used so that the
   jacobian and hessian may have fixed sized, which is much faster than dynamic
   sized.
   :param jacobian: The Jacobian matrix, with dimension 1 x in-degree.
   :param hessian: The Hessian matrix, with dimension in-degree x in-degree.
   :param d_grad1: The double type 1st order gradient
   :param d_grad2: The double type 2nd order gradient
  */
  template <class T1, class T2>
  void forward_gradient_scalarops(
      T1& jacobian,
      T2& hessian,
      double& d_grad1,
      double& d_grad2) const;

  /* Converts the 1x1 matrix value to a scalar value. */
  void to_scalar();

  /*** The following methods are valid only for *stochastic* nodes. ***/
  /*** They may eventually be moved to a specific subclass.         ***/

  virtual double log_prob() const {
    return 0;
  }

  virtual void log_prob_iid(
      const graph::NodeValue& /* value */,
      Eigen::MatrixXd& /* log_probs */) const {}

  /*
   Convenience creating a matrix for passing to log_prob_iid(value, matrix)
   and returning it.
   Returning this large object is fine because Eigen supports move semantics.
  */
  virtual Eigen::MatrixXd log_prob_iid(
      const graph::NodeValue& /* value */) const;

  /*
   Computes the first and second gradients of the log prob
   of this node with respect to a given target node and
   adds them to the passed-in gradient parameters.
   Note that for this computation to be correct,
   gradients (the grad1 and grad2 properties of nodes)
   must have been updated all the way from the
   target node to this node.
   This is because this method only performs a local computation
   and relies on the grad1 and grad2 attributes of nodes.
  */
  virtual void gradient_log_prob(
      const graph::Node* target_node,
      double& /* grad1 */,
      double& /* grad2 */) const {}

  virtual void gradient_log_prob(
      Eigen::MatrixXd& /* grad1 */,
      Eigen::MatrixXd& /* grad2_diag */) const {}
};

class ConstNode : public Node {
 public:
  explicit ConstNode(NodeValue value) : Node(NodeType::CONSTANT, value, {}) {}
  ~ConstNode() override {}
  bool is_stochastic() const override {
    return false;
  }
  bool is_mutable() const override {
    return false;
  }
  void eval(std::mt19937& /* */) override {
    throw std::runtime_error(
        "internal error: eval() should not be used for ConstNodes.");
  }
  std::unique_ptr<Node> clone() override {
    return std::make_unique<ConstNode>(value);
  }
  bool needs_gradient() const override {
    return false;
  }
  virtual std::string to_string() override {
    return "CONSTANT(" + value.to_string() + ")";
  }
};

// NOTE: the second kind of node -- Distribution is defined in distribution.h
// NOTE: the third kind of node -- Operator is defined in operator.h
// NOTE: the fourth kind of node -- Factor is defined in factor.h

using OrderedNodeIDs = std::set<NodeID>;
using Support = OrderedNodeIDs;
using MutableSupport = OrderedNodeIDs;

using DeterministicAffectedNodes = std::vector<NodeID>;
using StochasticAffectedNodes = std::vector<NodeID>;
using AffectedNodes =
    std::tuple<DeterministicAffectedNodes, StochasticAffectedNodes>;

class Graph {
 public:
  Graph() {}

  /*
  Copy constructor and copy assignment operator do not copy the inference
  results (if available) from the source graph.
  */
  Graph(const Graph& other);
  Graph& operator=(const Graph& other);

  ~Graph() {}
  std::string to_string() const;
  std::string to_dot() const;
  // Graph builder APIs -> return the node number
  NodeID add_constant(bool value);
  NodeID add_constant(double value);
  NodeID add_constant(natural_t value);
  NodeID add_constant(NodeValue value);
  NodeID add_constant_bool(bool value);
  NodeID add_constant_real(double value);
  NodeID add_constant_natural(natural_t value);
  NodeID add_constant_probability(double value);
  NodeID add_constant_pos_real(double value);
  NodeID add_constant_neg_real(double value);
  NodeID add_constant_bool_matrix(Eigen::MatrixXb& value);
  NodeID add_constant_real_matrix(Eigen::MatrixXd& value);
  NodeID add_constant_natural_matrix(Eigen::MatrixXn& value);
  NodeID add_constant_pos_matrix(Eigen::MatrixXd& value);
  NodeID add_constant_neg_matrix(Eigen::MatrixXd& value);
  NodeID add_constant_probability_matrix(Eigen::MatrixXd& value);
  NodeID add_constant_col_simplex_matrix(Eigen::MatrixXd& value);
  NodeID add_distribution(
      DistributionType dist_type,
      AtomicType sample_type,
      std::vector<NodeID> parents);
  NodeID add_distribution(
      DistributionType dist_type,
      ValueType sample_type,
      std::vector<NodeID> parents);
  NodeID add_operator(OperatorType op, std::vector<NodeID> parents);
  NodeID add_factor(FactorType fac_type, std::vector<NodeID> parents);
  // inference related
  void observe(NodeID var, bool val);
  void observe(NodeID var, double val);
  void observe(NodeID var, natural_t val);
  void observe(NodeID var, Eigen::MatrixXb& val);
  void observe(NodeID var, Eigen::MatrixXd& val);
  void observe(NodeID var, Eigen::MatrixXn& val);
  void observe(NodeID var, NodeValue val);
  /*
  Customize the type of transformation applied to a (set of)
  stochasitc node(s)
  :param transform_type: the type of transformation applied
  :param node_ids: the node ids that the transformation applies to
  */
  void customize_transformation(
      TransformType transform_type,
      std::vector<NodeID> node_ids);
  /*
  Removes all observations added to the graph.
  */
  void remove_observations();
  NodeID query(NodeID var); // returns the index of the query in the samples
  /*
  Draw Monte Carlo samples from the posterior distribution using a single chain.

  :param num_samples: The number of the MCMC samples.
  :param algorithm: The sampling algorithm, currently supporting REJECTION,
                    GIBBS, and NMC.
  :param seed: The seed provided to the random number generator.
  :returns: The posterior samples.
  */
  std::vector<std::vector<NodeValue>>&
  infer(uint num_samples, InferenceType algorithm, uint seed = 5123401);
  /*
  Draw Monte Carlo samples from the posterior distribution using multiple
  chains.

  :param num_samples: The number of the MCMC samples of each chain.
  :param algorithm: The sampling algorithm, currently supporting REJECTION,
                    GIBBS, and NMC.
  :param seed: The seed provided to the random number generator of the first
               chain.
  :param n_chains: The number of MCMC chains.
  :param infer_config: Other parameters for infer.
  :returns: The posterior samples from all chains.
  */
  std::vector<std::vector<std::vector<NodeValue>>>& infer(
      uint num_samples,
      InferenceType algorithm,
      uint seed,
      uint n_chains,
      InferConfig infer_config = InferConfig());
  /*
  Make point estimates of the posterior means from a single MCMC chain.
  :param num_samples: The number of the MCMC samples.
  :param algorithm: The sampling algorithm, currently supporting REJECTION,
  GIBBS, and NMC. :param seed: The seed provided to the random number generator.
  :returns: The posterior means.
  */
  std::vector<double>&
  infer_mean(uint num_samples, InferenceType algorithm, uint seed = 5123401);
  /*
  Make point estimates of the posterior means from multiple MCMC chains.

  :param num_samples: The number of the MCMC samples of each chain.
  :param algorithm: The sampling algorithm, currently supporting REJECTION,
                    GIBBS, and NMC.
  :param seed: The seed provided to the random number generator of the first
               chain.
  :param n_chains: The number of MCMC chains.
  :param infer_config: Other parameters for infer.
  :returns: The posterior means from all chains.
  */
  std::vector<std::vector<double>>& infer_mean(
      uint num_samples,
      InferenceType algorithm,
      uint seed,
      uint n_chains,
      InferConfig infer_config = InferConfig());
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
  std::vector<std::vector<double>>& variational(
      uint num_iters,
      uint steps_per_iter,
      uint seed = 5123401,
      uint elbo_samples = 0);
  std::vector<double>& get_elbo() {
    return elbo_vals;
  }

  /*
  The support of a graph includes *all* nodes, including
  distribution nodes and constant nodes, that are needed to determine the value
  of query and observed variables.
  This is a set of node ids *topologically ordered*,
  that is, if a support node A is an ancestor to support node B,
  then A appears before B in the support set.
  */
  Support compute_support();

  /*
  The *mutable* support is a subset of the support
  in which nodesare *mutable*, that is,
  they either have associated values or log probs that
  contribute to the joint probability based on their in-nodes,
  and need to be updated when a node affecting them changes value.
  Like the support set, this is also topologically ordered.
  As of October 2022, this means excluding constant and distribution nodes,
  because they have fixed values and do not *directly* contribute to the joint
  probability (distributions contribute through samples).
  */
  MutableSupport compute_mutable_support();

  Support _compute_support_given_mutable_choice(bool mutable_only);

  /*
  Given a node id N and a set of node ids S,
  computes node ids in S *affected* by the value of N.
  Set S must be topologically ordered.

  A node M is *affected* by N
  if M is a descendant of N and there is a path from N to M
  without any intermediary stochastic nodes.

  Intuitively, affected nodes are those whose value or log probability
  need to be updated once the value of N changes in order to keep
  the graph consistency.

  The affected nodes are returned in a pair of containers.
  The first one contains the deterministic affected nodes,
  and the second one contains the stochastic affected nodes.
  */
  AffectedNodes compute_affected_nodes(
      NodeID node_id,
      const OrderedNodeIDs& ordered_node_ids);

  AffectedNodes compute_affected_nodes_except_self(
      NodeID node_id,
      const OrderedNodeIDs& ordered_node_ids);

  AffectedNodes _compute_affected_nodes(
      NodeID node_id,
      const OrderedNodeIDs& ordered_node_ids,
      bool include_root_node);

  AffectedNodes _compute_affected_nodes(
      NodeID node_id,
      std::function<bool(NodeID descendant_id)> include);

  void eval_and_update_backgrad(const std::vector<Node*>& mutable_support);

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
      NodeID tgt_idx,
      NodeID src_idx,
      uint seed,
      NodeValue& value,
      double& grad1,
      double& grad2);
  /*
  Evaluate all nodes in the support and compute their gradients in
  backward mode. (used for unit tests) :param grad1: Output value of first
  gradient.
  :param seed: Random number generator seed.
  */
  void eval_and_grad(std::vector<DoubleMatrix*>& grad1, uint seed = 5123412);

  /*
  Compute the backward mode gradients for all nodes in the support
  (used for unit tests)
  :param grad1: Output value of first gradient.
  */
  void test_grad(std::vector<DoubleMatrix*>& grad1);

  /*
  Evaluate the deterministic descendants of the source node and compute
  the logprob_gradient of all stochastic descendants in the support
  including the source node.

  :param src_idx: The index of the node to evaluate the gradients w.r.t., must
                  be a vector valued node.
  :param grad1: Output value of first gradient (double)
  :param grad2: Output value of the second gradient (double)
  */
  void gradient_log_prob(NodeID src_idx, double& grad1, double& grad2);
  /*
  Evaluate the deterministic descendants of the source node and compute
  the sum of logprob of all stochastic descendants in the support
  including the source node.

  :param src_idx: source node
  :returns: The sum of log_prob of source node and all stochastic descendants.
  */
  double log_prob(NodeID src_idx);
  /*
  Evaluate the full log probability over the support of the graph.
  :returns: The sum of log_prob of stochastic nodes in the support.
  */
  double full_log_prob();
  std::vector<std::vector<double>>& get_log_prob();

  // TODO: This public method returns a pointer to an internal data structure
  // of the graph; this seems like a bad idea. We need it to be public though
  // because transform_test.cpp uses the node pointer to then obtain a pointer
  // to the value of the node and mutates that value to see what happens.
  // We should consider trying to find a safer way to test this functionality.
  Node* check_node(NodeID node_id, NodeType node_type);
  friend class GlobalState;
  // TODO: create Samples class and remove the following friend classes
  friend class GlobalMH;
  friend class GlobalMH;
  friend class RandomWalkMH;
  friend class HMC;

  void collect_performance_data(bool b);
  std::string performance_report();

  // private:
  // TODO: a lot of members used to be private, but we need access to them
  // (for example from the now external NMC class).
  // We need to rethink the use of Graph as a data
  // structure and the access to its members. Right now algorithms were put
  // "inside" Graph so they would have such access,
  // but Graph should be not depend on any algorithms,
  // so all this needs to be cleaned up.
  Node* check_observed_node(NodeID node_id, bool is_scalar);
  void add_observe(Node* node, NodeValue val);
  Node* get_node(NodeID node_id);
  void check_node_id(NodeID node_id);

  /*
  Adds node to graph, assuming it is already properly connected to parents,
  returning its new node id.
  */
  NodeID add_node(std::unique_ptr<Node> node);

  /* Clones given node and adds it to graph, returning its id. */
  NodeID duplicate(const std::unique_ptr<Node>& node) {
    return add_node(node->clone());
  }

  /*
  Remove node from the graph.
  This forces compacting of internal vectors in Graph which alter
  node ids, so the method returns a function mapping
  original node ids to new node ids.
  If the node is observed or in query, it stops being so.
  */
  std::function<NodeID(NodeID)> remove_node(NodeID node_id);
  std::function<NodeID(NodeID)> remove_node(std::unique_ptr<Node>& node);

  std::vector<Node*> convert_parent_ids(
      const std::vector<NodeID>& parents) const;
  std::vector<NodeID> get_parent_ids(
      const std::vector<Node*>& parent_nodes) const;
  void _infer(
      uint num_samples,
      InferenceType algorithm,
      uint seed,
      InferConfig infer_config);
  void _infer_parallel(
      uint num_samples,
      InferenceType algorithm,
      uint seed,
      uint n_chains,
      InferConfig infer_config);

  uint thread_index;
  std::vector<std::unique_ptr<Node>> nodes; // all nodes in topological order
  std::set<NodeID> observed; // set of observed nodes
  // we store redundant information in queries and queried. The latter is a
  // cache of the queried nodes while the former gives the order of nodes
  // queried
  std::vector<NodeID> queries; // list of queried node ids
  std::vector<std::vector<NodeValue>> samples;
  std::vector<std::vector<std::vector<NodeValue>>> samples_allchains;
  std::vector<double> means;
  std::vector<std::vector<double>> means_allchains;
  Graph* master_graph = nullptr;
  AggregationType agg_type;
  uint agg_samples;
  std::vector<std::vector<double>> variational_params;
  std::vector<double> elbo_vals;
  void collect_sample();
  void rejection(uint num_samples, uint seed, InferConfig infer_config);
  void gibbs(uint num_samples, uint seed, InferConfig infer_config);
  void nmc(uint num_samples, uint seed, InferConfig infer_config);
  void nuts(uint num_samples, uint seed, InferConfig infer_config);
  void cavi(
      uint num_iters,
      uint steps_per_iter,
      std::mt19937& gen,
      uint elbo_samples);

  // TODO: Review what members of this class can be made static.

  void collect_log_prob(double log_prob);
  std::vector<double> log_prob_vals;
  std::vector<std::vector<double>> log_prob_allchains;
  std::map<TransformType, std::unique_ptr<Transformation>>
      common_transformations;
  void _test_backgrad(
      MutableSupport& mutable_support,
      std::vector<DoubleMatrix*>& grad1);

  ProfilerData profiler_data;
  bool _collect_performance_data = false;
  std::string _performance_report;
  void _produce_performance_report(
      uint num_samples,
      InferenceType algorithm,
      uint seed);
  void pd_begin(ProfilerEvent kind) {
    if (_collect_performance_data) {
      profiler_data.begin(kind);
    }
  }
  void pd_finish(ProfilerEvent kind) {
    if (_collect_performance_data) {
      profiler_data.finish(kind);
    }
  }

  void reindex_nodes();

  // Every node in the graph has a value; when we propose a new graph state,
  // we update the values. If we then reject the proposed new state, we need
  // to restore the values. This vector stores the original values of the
  // nodes that we change during the proposal step.
  // We do the same for the log probability of the stochastic nodes
  // affected by the last revertible set and propagate operation
  // see (revertibly_set_and_propagate method).
 private:
  bool _old_values_vector_has_the_right_size = false;
  std::vector<NodeValue> _old_values;
  double _old_sto_affected_nodes_log_prob = 0;

  inline void _ensure_old_values_has_the_right_size() {
    if (not _old_values_vector_has_the_right_size) {
      _old_values = std::vector<NodeValue>(nodes.size());
      _old_values_vector_has_the_right_size = true;
    }
  }

  inline void _check_old_values_are_valid() {
    if (not _old_values_vector_has_the_right_size) {
      throw std::invalid_argument(
          "Old value requested but old values are invalid right now");
    }
  }

  // Members keeping graph structure information useful for evaluation
  // and inference.

  // We define getters for these properties that ensure they are up-to-date.
#define CACHED_PROPERTY(type, property, private_or_public) \
 private:                                                  \
  type _##property;                                        \
                                                           \
  private_or_public:                                       \
  const type& property() {                                 \
    _ensure_evaluation_and_inference_readiness();          \
    return _##property;                                    \
  }

#define CACHED_PUBLIC_PROPERTY(type, property) \
  CACHED_PROPERTY(type, property, public)

#define CACHED_PRIVATE_PROPERTY(type, property) \
  CACHED_PROPERTY(type, property, private)

  // A graph maintains of a vector of nodes; the index into that vector is
  // the id of the node. We often need to translate from node ids into node
  // pointers; to do so quickly we obtain the address of
  // every node in the graph up front and then look it up when we need it.
  CACHED_PUBLIC_PROPERTY(std::vector<Node*>, node_ptrs)

  // The set of mutable support nodes in the graph.
  // We keep both node ids and node pointer forms.
  CACHED_PUBLIC_PROPERTY(MutableSupport, mutable_support)
  CACHED_PUBLIC_PROPERTY(std::vector<Node*>, mutable_support_ptrs)

  // Nodes in mutable support that are not directly observed.
  // As usual, topologically ordered
  CACHED_PUBLIC_PROPERTY(std::vector<Node*>, unobserved_mutable_support)

  // Nodes in unobserved_mutable_support that are stochastic.
  // As usual, topologically ordered
  CACHED_PUBLIC_PROPERTY(std::vector<Node*>, unobserved_sto_mutable_support)

  // These vectors are the same size as unobserved_sto_mutable_support.
  // The i-th elements are vectors of nodes which are respectively
  // the vector of the immediate stochastic descendants of
  // node with index i in the
  // support, and
  // the vector of the intervening deterministic operator nodes
  // between the i-th node and its immediate stochastic descendants.

 private:
  CACHED_PRIVATE_PROPERTY(
      std::vector<std::vector<Node*>>,
      det_affected_operator_nodes)
  CACHED_PRIVATE_PROPERTY(std::vector<std::vector<Node*>>, sto_affected_nodes)

#undef CACHED_PROPERTY
#undef CACHED_PUBLIC_PROPERTY
#undef CACHED_PRIVATE_PROPERTY

  // Because unobserved_mutable_support and unobserved_sto_mutable_support do
  // not contain all nodes in the graph, it does not hold that a node id is
  // the same as its index in these vectors. The vectors below map node ids to
  // indices in unobserved_mutable_support and unobserved_sto_mutable_support
  // respectively.
  //
  // Note that, since not all nodes are in these vectors, some
  // elements of these index-mapping vectors should never be accessed.
  // That is, client code must be sure a node is
  // a support node (a stochastic support node, respectively)
  // before using the values in
  // unobserved_mutable_support_index_by_node_id
  // (unobserved_sto_mutable_support_index_by_node_id respectively).
 private:
  std::vector<size_t> unobserved_mutable_support_index_by_node_id;
  std::vector<size_t> unobserved_sto_mutable_support_index_by_node_id;

 private:
  bool ready_for_evaluation_and_inference = false;

  // Methods

  // Ensures graph is ready for evaluation and inference (by building
  // intermediate internal data structures).
  // The data structures are built only the first time the method is invoked.
  // After that, the method is simply ensuring they are built.
  // Note that this assumes the graph has not changed since the last
  // invocation.
  // If the graph does change, field "ready_for_evaluation_and_inference"
  // must be set to false.
  void _ensure_evaluation_and_inference_readiness() {
    if (not ready_for_evaluation_and_inference) {
      _compute_evaluation_and_inference_readiness_data();
      ready_for_evaluation_and_inference = true;
    }
  }

  void _compute_evaluation_and_inference_readiness_data();

  void _clear_evaluation_and_inference_readiness_data();

  void _collect_node_ptrs();

  void _collect_support();

  void _collect_affected_operator_nodes();

 public:
  void generate_sample();

  void collect_samples(uint num_samples, InferConfig infer_config);

  void collect_sample(InferConfig infer_config);

 public:
  const std::vector<Node*>& get_det_affected_operator_nodes(NodeID node_id);
  const std::vector<Node*>& get_sto_affected_nodes(NodeID node_id);

  inline const std::vector<Node*>& get_det_affected_operator_nodes(Node* node) {
    return get_det_affected_operator_nodes(node->index);
  }

  inline const std::vector<Node*>& get_sto_affected_nodes(Node* node) {
    return get_sto_affected_nodes(node->index);
  }

  // Sets a given node to a new value and
  // updates its deterministically affected nodes.
  // Does so in a revertible manner by saving old values and old stochastic
  // affected nodes log prob.
  // Old values can be accessed through get_old_* methods.
  // The reversion is executed by invoking revert_set_and_propagate.
  void revertibly_set_and_propagate(Node* node, const NodeValue& value);

  // Revert the last revertibly_set_and_propagate
  void revert_set_and_propagate(Node* node);

  void save_old_value(const Node* node);

  void save_old_values(const std::vector<Node*>& nodes);

  NodeValue& get_old_value(const Node* node);

  double get_old_sto_affected_nodes_log_prob() {
    return _old_sto_affected_nodes_log_prob;
  }

  void restore_old_value(Node* node);

  void restore_old_values(const std::vector<Node*>& det_nodes);

  void compute_gradients(const std::vector<Node*>& det_nodes);

  void eval(const std::vector<Node*>& det_nodes);

  void clear_gradients(Node* node);

  void clear_gradients(const std::vector<Node*>& nodes);

  void clear_gradients_of_node_and_its_affected_nodes(Node* node);

  double compute_log_prob_of(const std::vector<Node*>& sto_nodes);

  // Graph statistics
  std::string collect_statistics();

 private:
  class Statistics {
   public:
    explicit Statistics(Graph& g);
    std::string to_string();

   private:
    // 1. types
    using Counts_t = std::vector<uint>;
    using Matrix_t = std::vector<Counts_t>;
    using String_t = std::string;
    using Stream_t = std::ostringstream;

    // 2. Graph statistics
    uint num_edges; // Number of edges in the graph
    uint num_nodes; // Number of nodes in the graph
    uint max_in; // Largest number of incoming edges into a node
    uint max_out; // Largest number of outgoign edges from a node
    String_t graph_density; // From 0 to 1, 1 being a complete graph
    uint num_root_nodes; // Number of nodes with no incoming edges
    uint num_terminal_nodes; // Number of nodes with no outgoing edges

    void comp_graph_stats(Graph& g);
    String_t compute_density();
    void init_matrix(Matrix_t& matrix, uint rows, uint n_cols);

    // 3. Node statistics
    Counts_t dist_counts; // Counts of distribution types in dist. nodes
    Counts_t fact_counts; // Counts of factor types in factor nodes
    Counts_t node_type_counts; // Counts of node types in the graph
    Counts_t oper_counts; // Counts of operator types in operator nodes
    Matrix_t const_counts; // atomic type & var type
    Matrix_t root_terminal_per_node_type; // For each type, how many r & t

    void comp_node_stats(Graph& g);

    // 4. Edge statistics
    Counts_t in_edge_histogram; // Counts of incoming edges in graph
    Counts_t out_edge_histogram; // Counts of outgoing edges in graph
    Matrix_t in_edge_bytype; // incoming edges by node type
    Matrix_t out_edge_bytype; // outgoing edges by node type

    void comp_edge_stats(Graph& g);

    // 5. Reporting
    Stream_t report; // to hold the generated report
    uint tab;

    void gen_graph_stats_report();
    void gen_node_stats_report();
    void gen_edge_stats_report();
    void gen_edge_stats_report(String_t etype, Counts_t counts);
    void gen_operator_stats(Counts_t counts);
    void gen_distribution_stats(Counts_t counts);
    void gen_factor_stats(Counts_t counts);
    void gen_constant_stats(Matrix_t counts);
    void gen_roots_and_terminals(NodeID node_id);

    void emit(String_t output, char banner = '\0');
    void emit_tab(uint n);
  };
};

/*
Indicates whether two nodes are equal (same type and same in-nodes).
This ignores out-nodes and node index.
*/
bool are_equal(const graph::Node& node1, const graph::Node& node2);

template <typename NodePtr>
std::string in_nodes_string(const NodePtr& node) {
  using boost::adaptors::transformed;
  using boost::algorithm::join;
  return join(
      node->in_nodes | transformed([](const Node* node) {
        return std::to_string(node->index);
      }),
      ", ");
}

/*
Creates a copy of a subgraph in a graph.
More specifically,
given a *topologically ordered* set S of nodes in graph
for each node N in S:
  C <- clone N
  clone_of[N] <- C
  redirects each in-node I of C to clone_of[I]
  add C to graph
Note that it is important that S be topologically ordered
so that each in-node I of C that happens to be in S
will have already been cloned when it is redirected.
*/
void duplicate_subgraph(
    Graph& graph,
    const std::vector<Node*>& subgraph_ordered_nodes);

/* Returns a vector of Node * from a vector of node ids for given graph. */
std::vector<Node*> from_id_to_ptr(
    const Graph& graph,
    const std::vector<NodeID>& node_ids);

} // namespace graph
} // namespace beanmachine
