/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include "beanmachine/graph/double_matrix.h"
#include "beanmachine/graph/profiler.h"

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
  UNKNOWN = 0,
  SCALAR = 1,
  BROADCAST_MATRIX,
  COL_SIMPLEX_MATRIX,
};

enum class AtomicType {
  UNKNOWN = 0,
  BOOLEAN = 1,
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

typedef NATURAL_TYPE natural_t;

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
            throw std::invalid_argument(
                "Trying to copy an NodeValue of unknown type.");
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
  UNKNOWN = 0,
  SAMPLE = 1, // This is the ~ operator in models
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
  CHOICE,
  TO_INT,
  CHOLESKY,
  MATRIX_EXP,
  LOG_PROB,
};

enum class DistributionType {
  UNKNOWN = 0,
  TABULAR,
  BERNOULLI,
  BERNOULLI_NOISY_OR,
  BETA,
  BINOMIAL,
  DIRICHLET,
  FLAT,
  NORMAL,
  LOG_NORMAL,
  HALF_NORMAL,
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
};

enum class FactorType {
  UNKNOWN = 0,
  EXP_PRODUCT = 1,
};

enum class NodeType {
  UNKNOWN = 0,
  CONSTANT = 1,
  DISTRIBUTION = 2,
  OPERATOR = 3,
  FACTOR = 4,
  MAX = 5
};

enum class InferenceType { UNKNOWN = 0, REJECTION = 1, GIBBS, NMC };

enum class AggregationType { UNKNOWN = 0, NONE = 1, MEAN };

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

enum class TransformType { NONE = 0, LOG = 1 };

class Transformation {
 public:
  Transformation() : transform_type(TransformType::NONE) {}
  explicit Transformation(TransformType transform_type)
      : transform_type(transform_type) {}
  virtual ~Transformation() {}

  /*
  Overload the () to perform the variable transformation y=f(x) from the
  constrained value x to unconstrained y
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual void operator()(
      const NodeValue& /* constrained */,
      NodeValue& /* unconstrained */) {}
  /*
  Perform the inverse variable transformation x=f^{-1}(y) from the
  unconstrained value y to the original constrained x
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual void inverse(
      NodeValue& /* constrained */,
      const NodeValue& /* unconstrained */) {}
  /*
  Return the log of the absolute jacobian determinant:
    log |det(d x / d y)|
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual double log_abs_jacobian_determinant(
      const NodeValue& /* constrained */,
      const NodeValue& /* unconstrained */) {
    return 0;
  }
  /*
  Given the gradient of the joint log prob w.r.t x, update the value so
  that it is taken w.r.t y:
    back_grad = back_grad * dx / dy + d(log |det(d x / d y)|) / dy
  :param back_grad: the gradient w.r.t x
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual void unconstrained_gradient(
      DoubleMatrix& /* back_grad */,
      const NodeValue& /* constrained */,
      const NodeValue& /* unconstrained */) {}

  TransformType transform_type;
};

class Node {
 public:
  bool is_observed = false;
  NodeType node_type;
  uint index; // index in Graph::nodes
  std::vector<Node*> in_nodes;
  std::vector<Node*> out_nodes;
  std::vector<uint> det_anc; // deterministic (operator) ancestors
  std::vector<uint> sto_anc; // stochastic ancestors
  NodeValue value;
  double grad1;
  double grad2;
  Eigen::MatrixXd Grad1;
  Eigen::MatrixXd Grad2;
  DoubleMatrix back_grad1;

  virtual bool is_stochastic() const {
    return false;
  }
  // only valid for stochastic nodes
  // TODO: shouldn't we then restrict them to those classes? See below.
  virtual double log_prob() const {
    return 0;
  }
  virtual bool needs_gradient() const {
    return true;
  }
  // gradient_log_prob is also only valid for stochastic nodes
  // TODO: shouldn't we then restrict them to those classes? See above.
  // this function adds the gradients to the passed in gradients
  virtual void gradient_log_prob(
      const graph::Node* target_node,
      double& /* grad1 */,
      double& /* grad2 */) const {}
  virtual void gradient_log_prob(
      Eigen::MatrixXd& /* grad1 */,
      Eigen::MatrixXd& /* grad2_diag */) const {}
  Node() {}
  explicit Node(NodeType node_type)
      : node_type(node_type), grad1(0), grad2(0) {}
  Node(NodeType node_type, NodeValue value)
      : node_type(node_type), value(value), grad1(0), grad2(0) {}
  // evaluate the node and store the result in `value` if appropriate
  // eval may involve sampling and that's why we need the random number engine
  virtual void eval(std::mt19937& gen) = 0;

  // Computes the first and second gradients of this node
  // with respect to some (unspecified -- see below) variable.
  // More specifically, it uses the values stored in this node's input
  // nodes grad1 and grad2 fields (or Grad1 and Grad2 if they are matrices)
  // to compute the value of this node's own gradient fields
  // (again grad1, grad2 or Grad1 and Grad2).
  // Note that this method does *not* compute the gradient of
  // this node with respect to its inputs,
  // but with respect to some (possibly distant) variable.
  // The method is neutral regarding which variable this is,
  // and simply computes its own gradient with respect
  // to the same variable its input node gradients are
  // with respect to.
  // In the planned refactoring of BMG's autograd (as of May 2022)
  // this should be replaced by the more fundamental and modular function
  // computing a node's gradient with respect to its own inputs,
  // which should then be used as needed in
  // applications of the chain rule.
  virtual void compute_gradients() {}

  /*
  Gradient backward propagation: computes the 1st-order gradient update and
  add it to the parent's back_grad1.
  */
  virtual void backward() {}
  virtual ~Node() {}
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
  // Converts the 1x1 matrix value to a scalar value.
  void to_scalar();
};

class ConstNode : public Node {
 public:
  explicit ConstNode(NodeValue value) : Node(NodeType::CONSTANT, value) {}
  void eval(std::mt19937& /* unused */) override {}
  ~ConstNode() override {}
  bool needs_gradient() const override {
    return false;
  }
};

// NOTE: the second kind of node -- Distribution is defined in distribution.h
// NOTE: the third kind of node -- Operator is defined in operator.h
// NOTE: the fourth kind of node -- Factor is defined in factor.h

struct Graph {
  Graph() {}

  /*
  This copy constructor does not copy the inference results (if available)
  from the source graph.
  */
  Graph(const Graph& other);

  ~Graph() {}
  std::string to_string() const;
  std::string to_dot() const;
  // Graph builder APIs -> return the node number
  uint add_constant(bool value);
  uint add_constant(double value);
  uint add_constant(natural_t value);
  uint add_constant(NodeValue value);
  uint add_constant_probability(double value);
  uint add_constant_pos_real(double value);
  uint add_constant_neg_real(double value);
  uint add_constant_bool_matrix(Eigen::MatrixXb& value);
  uint add_constant_real_matrix(Eigen::MatrixXd& value);
  uint add_constant_natural_matrix(Eigen::MatrixXn& value);
  uint add_constant_pos_matrix(Eigen::MatrixXd& value);
  uint add_constant_neg_matrix(Eigen::MatrixXd& value);
  uint add_constant_probability_matrix(Eigen::MatrixXd& value);
  uint add_constant_col_simplex_matrix(Eigen::MatrixXd& value);
  uint add_distribution(
      DistributionType dist_type,
      AtomicType sample_type,
      std::vector<uint> parents);
  uint add_distribution(
      DistributionType dist_type,
      ValueType sample_type,
      std::vector<uint> parents);
  uint add_operator(OperatorType op, std::vector<uint> parents);
  uint add_factor(FactorType fac_type, std::vector<uint> parents);
  // inference related
  void observe(uint var, bool val);
  void observe(uint var, double val);
  void observe(uint var, natural_t val);
  void observe(uint var, Eigen::MatrixXb& val);
  void observe(uint var, Eigen::MatrixXd& val);
  void observe(uint var, Eigen::MatrixXn& val);
  void observe(uint var, NodeValue val);
  /*
  Customize the type of transformation applied to a (set of)
  stochasitc node(s)
  :param transform_type: the type of transformation applied
  :param node_ids: the node ids that the transformation applies to
  */
  void customize_transformation(
      TransformType transform_type,
      std::vector<uint> node_ids);
  /*
  Removes all observations added to the graph.
  */
  void remove_observations();
  uint query(uint var); // returns the index of the query in the samples
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
  The support of a graph is the set of operator and factor nodes that
  are needed to determine the value of query and observed variables. In other
  words, it is the set of queried and observed variables themselves plus their
  ancestors that are operator and factor nodes.
  */
  std::set<uint> compute_ordered_support_node_ids();
  /*
  The full support of a graph includes *all* nodes, including
  distribution nodes and constant nodes, that are needed to determine the value
  of query and observed variables
  */
  std::set<uint> compute_full_ordered_support_node_ids();

  std::set<uint> compute_ordered_support_node_ids_with_operators_only_choice(
      bool operator_factor_only);

  /*
  Computes the _affected nodes_ of a root node.

  Intuitively, these are the immediate, local descendants of the root node
  whose values or probabilities must be recalculated when
  the root node value changes.

  In a Bayesian network, which only contains stochastic nodes,
  the affected nodes would be the root node itself (since its probability
  changes according to its value) and its _children_ (whose probabilities
  also change since the root node is a parent and helps determining
  their probabilities).

  This is also essentially the case in BMG, but with one caveat:
  because BMG represents the deterministic computation of
  these children's distributions as explicit deterministic nodes in the graph,
  the nodes that would be the root node children in a Bayesian network are not
  root node children in BMG (since there are intervening deterministic
  nodes between the root note and these stochastic would-be children).
  For this reason, one needs to traverse these deterministic nodes
  until the stochastic would-be children are found.
  And because these deterministic nodes participate directly
  in the re-calculation of would-be children,
  they are also included in the set of affected nodes.

  Moreover, stochastic and deterministic
  affected nodes are returned in two separate collections
  since client code will often need to manipulate them very differently,
  typically re-computing the *values* of deterministic nodes,
  and re-computing the *probability* of stochastic nodes.

  The method guarantees to return the deterministic node in
  a topologically sorted order from the source.
  This ensures that evaluating these nodes individually
  in the given order produces the correct global result.

  :param node_id: the id (index in topological order) of the node for which
  we are computing the descendants
  :param ordered_support_node_ids: the (ordered) set of indices of the
  distribution support.
  :returns: vector of intervening operator deterministic
  nodes and vector of stochastic nodes that are operators and immediate
  stochastic descendants of the current node and in the support (that is to say,
  we don't return descendants of stochastic descendants). The current node is
  included in result if it is in support and is stochastic.
  */
  std::tuple<std::vector<uint>, std::vector<uint>> compute_affected_nodes(
      uint node_id,
      const std::set<uint>& ordered_support_node_ids);

  /*
  This function is almost the same as `compute_affected_nodes` above, with
  a few key differences:
  1. the deterministic nodes among the nodes returned by
`compute_affected_nodes` only include operator deterministic nodes, which have
  values which need to be re-calculated during inference. This function returns
  *all* the deterministic nodes between the current node and its stochastic
  children, including distribution nodes, constants, etc
  2. `compute_affected_nodes` includes the current stochastic node, while this
  function only includes its children
  :param node_id: the id (index in topological order) of the node for which we
  are computing the descendants
  :param ordered_support_node_ids: the set of indices of the distribution
  support.
  :returns: vector of all intermediate deterministic nodes and vector of
  stochastic nodes and immediate stochastic descendants of the current node and
  in the support (that is to say, we don't return descendants of stochastic
  descendants). The current node is included in result if it is in support and
  is stochastic.
  */
  std::tuple<std::vector<uint>, std::vector<uint>> compute_children(
      uint node_id,
      const std::set<uint>& ordered_support_node_ids);

  std::tuple<std::vector<uint>, std::vector<uint>>
  _compute_nodes_until_stochastic(
      uint node_id,
      const std::set<uint>& ordered_support_node_ids,
      bool affected_only,
      bool include_root_node);

  std::tuple<std::vector<uint>, std::vector<uint>> compute_ancestors(
      uint node_id);

  void eval_and_update_backgrad(std::vector<Node*>& ordered_supp);
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
      uint tgt_idx,
      uint src_idx,
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
  void gradient_log_prob(uint src_idx, double& grad1, double& grad2);
  /*
  Evaluate the deterministic descendants of the source node and compute
  the sum of logprob of all stochastic descendants in the support
  including the source node.

  :param src_idx: source node
  :returns: The sum of log_prob of source node and all stochastic descendants.
  */
  double log_prob(uint src_idx);
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
  Node* check_node(uint node_id, NodeType node_type);
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
  Node* check_observed_node(uint node_id, bool is_scalar);
  void add_observe(Node* node, NodeValue val);
  Node* get_node(uint node_id);
  void check_node_id(uint node_id);

  uint add_node(std::unique_ptr<Node> node, std::vector<uint> parents);
  std::vector<Node*> convert_parent_ids(const std::vector<uint>& parents) const;
  std::vector<uint> get_parent_ids(
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
  std::set<uint> observed; // set of observed nodes
  // we store redundant information in queries and queried. The latter is a
  // cache of the queried nodes while the former gives the order of nodes
  // queried
  std::vector<uint> queries; // list of queried node ids
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
  void cavi(
      uint num_iters,
      uint steps_per_iter,
      std::mt19937& gen,
      uint elbo_samples);
  /*
  Evaluate the full log probability over the support of the graph.
  :param ordered_supp: node pointers in the support in topological
  order.
  :returns: The sum of log_prob of stochastic nodes in the
  support.
  */

  // TODO: Review what members of this class can be made static.

  void collect_log_prob(double log_prob);
  std::vector<double> log_prob_vals;
  std::vector<std::vector<double>> log_prob_allchains;
  std::map<TransformType, std::unique_ptr<Transformation>>
      common_transformations;
  void _test_backgrad(
      std::set<uint>& ordered_support_node_ids,
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

  // members brought in from MH class since they are really Graph properties
 public:
  // A graph maintains of a vector of nodes; the index into that vector is
  // the id of the node. We often need to translate from node ids into node
  // pointers; to do so quickly we obtain the address of
  // every node in the graph up front and then look it up when we need it.
  std::vector<Node*> node_ptrs;

  // Every node in the graph has a value; when we propose a new graph state,
  // we update the values. If we then reject the proposed new state, we need
  // to restore the values. This vector stores the original values of the
  // nodes that we change during the proposal step.
  // We do the same for the log probability of the stochastic nodes
  // affected by the last revertible set and propagate operation
  // see (revertibly_set_and_propagate method).
  std::vector<NodeValue> old_values;
  double old_sto_affected_nodes_log_prob;

  // The support is the set of all nodes in the graph that are queried or
  // observed, directly or indirectly. We keep both node ids and node pointer
  // forms.
  std::set<uint> supp_ids;
  std::vector<Node*> supp;

  // Nodes in support that are not directly observed. Note that
  // the order of nodes in this vector matters! We must enumerate
  // them in order from lowest node identifier to highest.
  std::vector<Node*> unobserved_supp;

  // Nodes in unobserved_supp that are stochastic; similarly, order matters.
  std::vector<Node*> unobserved_sto_supp;

  // A vector containing the index of a node in vector unobserved_sto_supp for
  // each node_id. Since not all nodes are in unobserved_sto_support, some
  // elements of this vector should never be accessed.
  std::vector<uint> unobserved_sto_support_index_by_node_id;

  // These vectors are the same size as unobserved_sto_support.
  // The i-th elements are vectors of nodes which are
  // respectively the vector of
  // the immediate stochastic descendants of node with index i in the
  // support, and the vector of the intervening deterministic nodes
  // between the i-th node and its immediate stochastic descendants.
  // In other words, these are the cached results of
  // invoking graph::compute_affected_nodes
  // for each node.
  std::vector<std::vector<Node*>> sto_affected_nodes;
  std::vector<std::vector<Node*>> det_affected_nodes;

  bool ready_for_evaluation_and_inference = false;

  // Methods

  // Ensures graph is ready for evaluation and inference (by building
  // intermediate internal data structures).
  // The data structures are built only the first time the method is invoked.
  // After that, the method is simply ensuring they are built.
  // Note that this assumes the graph has not changed since the last invocation.
  // If the graph does change, client code can set field "ready" to false
  // and then invoke this method.
  void ensure_evaluation_and_inference_readiness();

  void collect_node_ptrs();

  void compute_support();

  void ensure_all_nodes_are_supported();

  void compute_initial_values();

  void compute_affected_nodes();

  void generate_sample();

  void collect_samples(uint num_samples, InferConfig infer_config);

  void collect_sample(InferConfig infer_config);

  const std::vector<Node*>& get_det_affected_nodes(Node* node);

  const std::vector<Node*>& get_sto_affected_nodes(Node* node);

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
    return old_sto_affected_nodes_log_prob;
  }

  void restore_old_value(Node* node);

  void restore_old_values(const std::vector<Node*>& det_nodes);

  void compute_gradients(const std::vector<Node*>& det_nodes);

  void eval(const std::vector<Node*>& det_nodes);

  void clear_gradients(Node* node);

  void clear_gradients(const std::vector<Node*>& nodes);

  void clear_gradients_of_node_and_its_affected_nodes(Node* node);

  double compute_log_prob_of(const std::vector<Node*>& sto_nodes);
};

} // namespace graph
} // namespace beanmachine
