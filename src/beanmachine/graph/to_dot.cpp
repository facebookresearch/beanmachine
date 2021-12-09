/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace graph {

class DOT {
 public:
  std::string str() {
    return os.str();
  }

  void start_graph() {
    os << "digraph \"graph\" {\n";
  }

  void end_graph() {
    os << "}\n";
  }

  void node(uint n, std::string label) {
    os << "  N" << n << "[label=\"" << label << "\"];\n";
  }

  void scalar(NodeValue v) {
    switch (v.type.atomic_type) {
      case AtomicType::BOOLEAN:
        os << v._bool;
        break;
      case AtomicType::NATURAL:
        os << v._natural;
        break;
      case AtomicType::REAL:
      case AtomicType::POS_REAL:
      case AtomicType::NEG_REAL:
      case AtomicType::PROBABILITY:
        os << v._double;
        break;
      default:
        os << "Scalar";
        break;
    }
  }

  void broadcast(NodeValue v) {
    // TODO: Better display of matrices
    os << "matrix";
  }

  void simplex(NodeValue v) {
    // TODO: Better display of simplexes
    os << "simplex";
  }

  void value(NodeValue v) {
    switch (v.type.variable_type) {
      case VariableType::SCALAR:
        scalar(v);
        break;
      case VariableType::BROADCAST_MATRIX:
        broadcast(v);
        break;
      case VariableType::COL_SIMPLEX_MATRIX:
        simplex(v);
        break;
      default:
        os << "Value";
    }
  }

  void constant(uint n, NodeValue v) {
    os << "  N" << n << "[label=\"";
    value(v);
    os << "\"];\n";
  }

  std::string distribution(DistributionType d) {
    switch (d) {
      case DistributionType::TABULAR:
        return "Tabular";
      case DistributionType::BERNOULLI:
        return "Bernoulli";
      case DistributionType::BERNOULLI_NOISY_OR:
        return "BernoulliNO";
      case DistributionType::BETA:
        return "Beta";
      case DistributionType::BINOMIAL:
        return "Binomial";
      case DistributionType::DIRICHLET:
        return "Dirichlet";
      case DistributionType::FLAT:
        return "Flat";
      case DistributionType::NORMAL:
        return "Normal";
      case DistributionType::HALF_CAUCHY:
        return "HalfCauchy";
      case DistributionType::STUDENT_T:
        return "StudentT";
      case DistributionType::BERNOULLI_LOGIT:
        return "BernoulliLogit";
      case DistributionType::GAMMA:
        return "Gamma";
      case DistributionType::BIMIXTURE:
        return "Bimixture";
      case DistributionType::CATEGORICAL:
        return "Categorical";
      case DistributionType::HALF_NORMAL:
        return "HalfNormal";
      default:
        return "distribution";
    }
  }

  void distribution(uint n, DistributionType d) {
    node(n, distribution(d));
  }

  std::string op(OperatorType o) {
    switch (o) {
      case OperatorType::SAMPLE:
      case OperatorType::IID_SAMPLE:
        return "~";
      case OperatorType::TO_INT:
        return "ToInt";
      case OperatorType::TO_REAL:
      case OperatorType::TO_REAL_MATRIX:
        return "ToReal";
      case OperatorType::TO_POS_REAL:
      case OperatorType::TO_POS_REAL_MATRIX:
        return "ToPosReal";
      case OperatorType::COMPLEMENT:
        return "Complement";
      case OperatorType::NEGATE:
        return "Negate";
      case OperatorType::EXP:
        return "exp";
      case OperatorType::EXPM1:
        return "expm1";
      case OperatorType::MULTIPLY:
        return "*";
      case OperatorType::ADD:
        return "+";
      case OperatorType::PHI:
        return "Phi";
      case OperatorType::LOGISTIC:
        return "Logistic";
      case OperatorType::IF_THEN_ELSE:
        return "IfThenElse";
      case OperatorType::CHOICE:
        return "Choice";
      case OperatorType::LOG1PEXP:
        return "Log1pExp";
      case OperatorType::LOGSUMEXP:
        return "LogSumExp";
      case OperatorType::LOGSUMEXP_VECTOR:
        return "LogSumExp";
      case OperatorType::LOG:
        return "Log";
      case OperatorType::POW:
        return "^";
      case OperatorType::LOG1MEXP:
        return "Log1mExp";
      case OperatorType::MATRIX_MULTIPLY:
        return "MatrixMultiply";
      case OperatorType::MATRIX_SCALE:
        return "MatrixScale";
      case OperatorType::TO_PROBABILITY:
        return "ToProb";
      case OperatorType::TO_NEG_REAL:
        return "ToNegReal";
      case OperatorType::INDEX:
        return "Index";
      case OperatorType::TO_MATRIX:
        return "ToMatrix";
      case OperatorType::COLUMN_INDEX:
        return "ColumnIndex";
      default:
        return "Operator";
    }
  }

  void op(uint n, OperatorType o) {
    node(n, op(o));
  }

  void edge(uint start_n, uint end_n) {
    os << "  N" << start_n << " -> N" << end_n << ";\n";
  }

  void observation(uint o, uint n) {
    // TODO: Include value of observation
    os << "  O" << o << "[label=\"Observation\"];\n";
    os << "  N" << n << " -> O" << o << ";\n";
  }

  void query(uint q, uint n) {
    os << "  Q" << q << "[label=\"Query\"];\n";
    os << "  N" << n << " -> Q" << q << ";\n";
  }

 private:
  std::ostringstream os;
};

std::string Graph::to_dot() const {
  DOT dot;
  dot.start_graph();

  // Nodes
  for (auto const& n : nodes) {
    Node* node = n.get();
    switch (node->node_type) {
      case NodeType::CONSTANT:
        dot.constant(node->index, node->value);
        break;
      case NodeType::DISTRIBUTION:
        dot.distribution(
            node->index,
            static_cast<const distribution::Distribution*>(node)->dist_type);
        break;
      case NodeType::OPERATOR:
        dot.op(node->index, static_cast<const oper::Operator*>(node)->op_type);
        break;
      case NodeType::FACTOR:
        // TODO: Better display of factors.
        dot.node(node->index, "Factor");
        break;
      default:
        dot.node(node->index, "Node");
        break;
    }
  }

  // Edges
  for (auto const& node : nodes) {
    for (Node* child : node->out_nodes) {
      dot.edge(node->index, child->index);
    }
  }

  // Observations
  uint obs = 0;
  for (uint id : observed) {
    dot.observation(obs, id);
    obs += 1;
  }

  // Queries
  for (uint q = 0; q < queries.size(); q += 1) {
    dot.query(q, queries[q]);
  }
  dot.end_graph();
  return dot.str();
}

} // namespace graph
} // namespace beanmachine
