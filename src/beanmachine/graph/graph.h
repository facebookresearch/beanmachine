// Copyright (c) Facebook, Inc. and its affiliates.
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
#include <vector>

#define NATURAL_TYPE unsigned long long int

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
            value.rows(),
            value.cols())),
        _matrix(value) {}
  explicit NodeValue(Eigen::MatrixXb& value)
      : type(ValueType(
            VariableType::BROADCAST_MATRIX,
            AtomicType::BOOLEAN,
            value.rows(),
            value.cols())),
        _bmatrix(value) {}
  explicit NodeValue(Eigen::MatrixXn& value)
      : type(ValueType(
            VariableType::BROADCAST_MATRIX,
            AtomicType::NATURAL,
            value.rows(),
            value.cols())),
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
            value.rows(),
            value.cols())),
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
    if (type.variable_type == VariableType::SCALAR) {
      switch (type.atomic_type) {
        case AtomicType::UNKNOWN: {
          throw std::invalid_argument(
              "Trying to copy an NodeValue of unknown type.");
        }
        case AtomicType::BOOLEAN: {
          _bool = other._bool;
          break;
        }
        case AtomicType::NATURAL: {
          _natural = other._natural;
          break;
        }
        default: {
          _double = other._double;
          break;
        }
      }
    } else if (type.variable_type == VariableType::BROADCAST_MATRIX) {
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
    } else if (type.variable_type == VariableType::COL_SIMPLEX_MATRIX) {
      _matrix = other._matrix;
    } else {
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
  EXP,
  EXPM1,
  MULTIPLY,
  ADD,
  PHI,
  LOGISTIC,
  IF_THEN_ELSE,
  LOG1PEXP,
  LOGSUMEXP,
  LOG,
  POW,
  LOG1MEXP,
  MATRIX_MULTIPLY,
  TO_PROBABILITY,
  INDEX,
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
  HALF_CAUCHY,
  STUDENT_T,
  BERNOULLI_LOGIT,
  GAMMA,
  BIMIXTURE,
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
  MAX
};

enum class InferenceType { UNKNOWN = 0, REJECTION = 1, GIBBS, NMC };

enum class AggregationType { UNKNOWN = 0, NONE = 1, MEAN };

struct DoubleMatrix {
  double _double;
  Eigen::MatrixXd _matrix;
};

struct InferConfig {
  bool keep_log_prob;
  double path_length;
  double step_size;

  ~InferConfig() {}
  InferConfig(
      bool keep_log_prob = false,
      double path_length = 1.0,
      double step_size = 1.0)
      : keep_log_prob(keep_log_prob),
        path_length(path_length),
        step_size(step_size) {}
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
  virtual double log_prob() const {
    return 0;
  }
  virtual bool needs_gradient() const {
    return true;
  }
  // gradient_log_prob is also only valid for stochastic nodes
  // this function adds the gradients to the passed in gradients
  virtual void gradient_log_prob(double& /* grad1 */, double& /* grad2 */)
      const {}
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
  // populate the derivatives
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
  Graph(const Graph& other);
  ~Graph() {}
  std::string to_string() const;
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
  std::set<uint> compute_support();
  std::tuple<std::vector<uint>, std::vector<uint>> compute_descendants(
      uint node_id,
      const std::set<uint>& support);
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
      uint tgt_idx,
      uint src_idx,
      uint seed,
      NodeValue& value,
      double& grad1,
      double& grad2);
  /*
  Evaluate all nodes in the support and compute their gradients in backward
  mode. (used for unit tests)
  :param grad1: Output value of first gradient.
  :param seed: Random number generator seed.
  */
  void eval_and_grad(std::vector<DoubleMatrix*>& grad1, uint seed = 5123412);
  /*
  Evaluate the deterministic descendants of the source node and compute
  the logprob_gradient of all stochastic descendants in the support including
  the source node.

  :param src_idx: The index of the node to evaluate the gradients w.r.t., must
                  be a vector valued node.
  :param grad1: Output value of first gradient (double), or gradient vector
                (Eigen::MatrixXd)
  :param grad2: Output value of the second gradient (double), or the diagonal
                terms of the gradient matrix (Eigen::MatrixXd).
  */
  template <class T>
  void gradient_log_prob(uint src_idx, T& grad1, T& grad2);
  /*
  Evaluate the deterministic descendants of the source node and compute
  the sum of logprob of all stochastic descendants in the support including
  the source node.

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
  Node* check_node(uint node_id, NodeType node_type);
  uint thread_index;

  void collect_performance_data(bool b);
  std::string performance_report();

 private:
  uint add_node(std::unique_ptr<Node> node, std::vector<uint> parents);
  std::vector<Node*> convert_parent_ids(const std::vector<uint>& parents) const;
  std::vector<uint> get_parent_ids(
      const std::vector<Node*>& parent_nodes) const;
  void _infer(uint num_samples, InferenceType algorithm, uint seed);
  void _infer_parallel(
      uint num_samples,
      InferenceType algorithm,
      uint seed,
      uint n_chains);
  std::vector<std::unique_ptr<Node>> nodes; // all nodes in topological order
  std::set<uint> observed; // set of observed nodes
  // we store redundant information in queries and queried. The latter is a
  // cache of the queried nodes while the former gives the order of nodes
  // queried
  std::vector<uint> queries; // list of queried nodenums
  std::set<uint> queried; // set of queried nodes
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
  void rejection(uint num_samples, std::mt19937& gen);
  void gibbs(uint num_samples, std::mt19937& gen);
  void nmc(uint num_samples, std::mt19937& gen);
  void cavi(
      uint num_iters,
      uint steps_per_iter,
      std::mt19937& gen,
      uint elbo_samples);
  InferConfig infer_config;
  /*
  Evaluate the full log probability over the support of the graph.
  :param ordered_supp: node pointers in the support in topological order.
  :returns: The sum of log_prob of stochastic nodes in the support.
  */
  double _full_log_prob(std::vector<Node*>& ordered_supp);
  void collect_log_prob(double log_prob);
  std::vector<double> log_prob_vals;
  std::vector<std::vector<double>> log_prob_allchains;
  std::map<TransformType, std::unique_ptr<Transformation>>
      common_transformations;

  bool _collect_performance_data = false;
  std::string _performance_report;
  void _produce_performance_report();
};

/*
A temporary solution to use NMC sampler on Dirichlet sample.
*/
void nmc_step_for_dirichlet(
    Node* tgt_node,
    const std::vector<uint>& det_nodes,
    const std::vector<uint>& sto_nodes,
    const std::vector<Node*>& node_ptrs,
    std::vector<NodeValue>& old_values,
    std::mt19937& gen);

} // namespace graph
} // namespace beanmachine
