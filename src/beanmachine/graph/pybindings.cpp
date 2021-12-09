/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/pybindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace beanmachine {
namespace graph {

namespace py = pybind11;

PYBIND11_MODULE(graph, module) {
  module.doc() = "module for python bindings to the graph API";

  py::enum_<TransformType>(module, "TransformType")
      .value("NONE", TransformType::NONE)
      .value("LOG", TransformType::LOG);

  py::enum_<InitType>(module, "InitType")
      .value("PRIOR", InitType::PRIOR)
      .value("RANDOM", InitType::RANDOM)
      .value("ZERO", InitType::ZERO);

  py::enum_<VariableType>(module, "VariableType")
      .value("SCALAR", VariableType::SCALAR)
      .value("BROADCAST_MATRIX", VariableType::BROADCAST_MATRIX)
      .value("COL_SIMPLEX_MATRIX", VariableType::COL_SIMPLEX_MATRIX);

  py::enum_<AtomicType>(module, "AtomicType")
      .value("BOOLEAN", AtomicType::BOOLEAN)
      .value("PROBABILITY", AtomicType::PROBABILITY)
      .value("REAL", AtomicType::REAL)
      .value("POS_REAL", AtomicType::POS_REAL)
      .value("NATURAL", AtomicType::NATURAL)
      .value("NEG_REAL", AtomicType::NEG_REAL);

  py::class_<ValueType>(module, "ValueType")
      .def(py::init<VariableType, AtomicType, uint, uint>())
      .def(
          "to_string",
          &ValueType::to_string,
          "string representation of the type");

  py::class_<NodeValue>(module, "NodeValue")
      .def(py::init<bool>())
      .def(py::init<double>())
      .def(py::init<graph::natural_t>())
      .def(py::init<Eigen::MatrixXb&>())
      .def(py::init<Eigen::MatrixXd&>());

  py::enum_<OperatorType>(module, "OperatorType")
      .value("SAMPLE", OperatorType::SAMPLE)
      .value("TO_INT", OperatorType::TO_INT)
      .value("TO_REAL", OperatorType::TO_REAL)
      .value("TO_POS_REAL", OperatorType::TO_POS_REAL)
      .value("COMPLEMENT", OperatorType::COMPLEMENT)
      .value("NEGATE", OperatorType::NEGATE)
      .value("EXP", OperatorType::EXP)
      .value("EXPM1", OperatorType::EXPM1)
      .value("MULTIPLY", OperatorType::MULTIPLY)
      .value("ADD", OperatorType::ADD)
      .value("PHI", OperatorType::PHI)
      .value("LOGISTIC", OperatorType::LOGISTIC)
      .value("LOG1PEXP", OperatorType::LOG1PEXP)
      .value("LOG1MEXP", OperatorType::LOG1MEXP)
      .value("LOGSUMEXP", OperatorType::LOGSUMEXP)
      .value("IF_THEN_ELSE", OperatorType::IF_THEN_ELSE)
      .value("LOG", OperatorType::LOG)
      .value("POW", OperatorType::POW)
      .value("MATRIX_MULTIPLY", OperatorType::MATRIX_MULTIPLY)
      .value("MATRIX_SCALE", OperatorType::MATRIX_SCALE)
      .value("TO_PROBABILITY", OperatorType::TO_PROBABILITY)
      .value("INDEX", OperatorType::INDEX)
      .value("BROADCAST_ADD", OperatorType::BROADCAST_ADD)
      .value("TO_MATRIX", OperatorType::TO_MATRIX)
      .value("LOGSUMEXP_VECTOR", OperatorType::LOGSUMEXP_VECTOR)
      .value("COLUMN_INDEX", OperatorType::COLUMN_INDEX)
      .value("TO_REAL_MATRIX", OperatorType::TO_REAL_MATRIX)
      .value("TO_POS_REAL_MATRIX", OperatorType::TO_POS_REAL_MATRIX)
      .value("TO_NEG_REAL", OperatorType::TO_NEG_REAL)
      .value("CHOICE", OperatorType::CHOICE);

  py::enum_<DistributionType>(module, "DistributionType")
      .value("TABULAR", DistributionType::TABULAR)
      .value("BERNOULLI", DistributionType::BERNOULLI)
      .value("BERNOULLI_NOISY_OR", DistributionType::BERNOULLI_NOISY_OR)
      .value("BETA", DistributionType::BETA)
      .value("BINOMIAL", DistributionType::BINOMIAL)
      .value("FLAT", DistributionType::FLAT)
      .value("NORMAL", DistributionType::NORMAL)
      .value("HALF_NORMAL", DistributionType::HALF_NORMAL)
      .value("HALF_CAUCHY", DistributionType::HALF_CAUCHY)
      .value("STUDENT_T", DistributionType::STUDENT_T)
      .value("BERNOULLI_LOGIT", DistributionType::BERNOULLI_LOGIT)
      .value("GAMMA", DistributionType::GAMMA)
      .value("BIMIXTURE", DistributionType::BIMIXTURE)
      .value("DIRICHLET", DistributionType::DIRICHLET)
      .value("CATEGORICAL", DistributionType::CATEGORICAL);

  py::enum_<FactorType>(module, "FactorType")
      .value("EXP_PRODUCT", FactorType::EXP_PRODUCT);

  py::enum_<NodeType>(module, "NodeType")
      .value("CONSTANT", NodeType::CONSTANT)
      .value("DISTRIBUTION", NodeType::DISTRIBUTION)
      .value("OPERATOR", NodeType::OPERATOR)
      .value("FACTOR", NodeType::FACTOR);

  py::enum_<InferenceType>(module, "InferenceType")
      .value("REJECTION", InferenceType::REJECTION)
      .value("GIBBS", InferenceType::GIBBS)
      .value("NMC", InferenceType::NMC);

  py::class_<Node>(module, "Node");

  py::class_<InferConfig>(module, "InferConfig")
      .def(py::init())
      .def(py::init<bool, double, double, uint, bool>())
      .def_readwrite("keep_log_prob", &InferConfig::keep_log_prob)
      .def_readwrite("path_length", &InferConfig::path_length)
      .def_readwrite("step_size", &InferConfig::step_size)
      .def_readwrite("num_warmup", &InferConfig::num_warmup)
      .def_readwrite("keep_warmup", &InferConfig::keep_warmup);

  // CONSIDER: Remove the overloaded add_constant APIs; the overloaded API's
  // binding behaviour is a little confusing. For example,
  // add_constant(tensor(2.5)) has the effect of calling add_constant(True).
  py::class_<Graph>(module, "Graph")
      .def(py::init())
      .def("to_string", &Graph::to_string, "string representation of the graph")
      .def("to_dot", &Graph::to_dot, "DOT representation of the graph")
      .def(
          "add_constant_bool",
          (uint(Graph::*)(bool)) & Graph::add_constant,
          "add a Node with a constant boolean value",
          py::arg("value"))
      .def(
          "add_constant_real",
          (uint(Graph::*)(double)) & Graph::add_constant,
          "add a Node with a constant real value",
          py::arg("value"))
      .def(
          "add_constant_natural",
          (uint(Graph::*)(graph::natural_t)) & Graph::add_constant,
          "add a Node with a constant natural (integers >= 0) value",
          py::arg("value"))
      .def(
          "add_constant",
          (uint(Graph::*)(bool)) & Graph::add_constant,
          "add a Node with a constant boolean value",
          py::arg("value"))
      .def(
          "add_constant",
          (uint(Graph::*)(double)) & Graph::add_constant,
          "add a Node with a constant real value",
          py::arg("value"))
      .def(
          "add_constant",
          (uint(Graph::*)(graph::natural_t)) & Graph::add_constant,
          "add a Node with a constant natural (integers >= 0) value",
          py::arg("value"))
      .def(
          "add_constant",
          (uint(Graph::*)(NodeValue)) & Graph::add_constant,
          "add a Node with a constant value",
          py::arg("value"))
      .def(
          "add_constant_probability",
          (uint(Graph::*)(double)) & Graph::add_constant_probability,
          "add a Node with a constant probability value",
          py::arg("value"))
      .def(
          "add_constant_pos_real",
          (uint(Graph::*)(double)) & Graph::add_constant_pos_real,
          "add a Node with a constant positive real (>=0) value",
          py::arg("value"))
      .def(
          "add_constant_neg_real",
          (uint(Graph::*)(double)) & Graph::add_constant_neg_real,
          "add a Node with a constant negative real (<=0) value",
          py::arg("value"))
      .def(
          "add_constant_real_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) & Graph::add_constant_real_matrix,
          "add a Node with a constant real-valued matrix",
          py::arg("value"))
      .def(
          "add_constant_bool_matrix",
          (uint(Graph::*)(Eigen::MatrixXb&)) & Graph::add_constant_bool_matrix,
          "add a Node with a constant Boolean-valued matrix",
          py::arg("value"))
      .def(
          "add_constant_natural_matrix",
          (uint(Graph::*)(Eigen::MatrixXn&)) &
              Graph::add_constant_natural_matrix,
          "add a Node with a constant natural-valued matrix",
          py::arg("value"))
      .def(
          "add_constant_pos_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) & Graph::add_constant_pos_matrix,
          "add a Node with a constant element-wise positive valued matrix",
          py::arg("value"))
      .def(
          "add_constant_neg_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) & Graph::add_constant_neg_matrix,
          "add a Node with a constant element-wise negative valued matrix",
          py::arg("value"))
      .def(
          "add_constant_col_simplex_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) &
              Graph::add_constant_col_simplex_matrix,
          "add a Node with a constant matrix with each column a simplex",
          py::arg("value"))
      .def(
          "add_constant_probability_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) &
              Graph::add_constant_probability_matrix,
          "add a Node with a constant probability-valued matrix",
          py::arg("value"))
      .def(
          "add_distribution",
          (uint(Graph::*)(DistributionType, AtomicType, std::vector<uint>)) &
              Graph::add_distribution,
          "add a probability distribution Node",
          py::arg("dist_type"),
          py::arg("sample_type"),
          py::arg("parents"))
      .def(
          "add_distribution",
          (uint(Graph::*)(DistributionType, ValueType, std::vector<uint>)) &
              Graph::add_distribution,
          "add a probability distribution Node",
          py::arg("dist_type"),
          py::arg("sample_type"),
          py::arg("parents"))
      .def(
          "add_operator",
          &Graph::add_operator,
          "add an operator Node",
          py::arg("op"),
          py::arg("parents"))
      .def(
          "add_factor",
          &Graph::add_factor,
          "add a factor Node",
          py::arg("fac_type"),
          py::arg("parents"))
      .def(
          "observe",
          (void (Graph::*)(uint, bool)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "observe",
          (void (Graph::*)(uint, double)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "observe",
          (void (Graph::*)(uint, natural_t)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "observe",
          (void (Graph::*)(uint, Eigen::MatrixXd&)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "observe",
          (void (Graph::*)(uint, Eigen::MatrixXb&)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "observe",
          (void (Graph::*)(uint, Eigen::MatrixXn&)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "observe",
          (void (Graph::*)(uint, NodeValue)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
      .def(
          "remove_observations",
          (void (Graph::*)()) & Graph::remove_observations,
          "remove all observations from the graph")
      .def("query", &Graph::query, "query a node", py::arg("node_id"))
      .def(
          "infer_mean",
          (std::vector<double> & (Graph::*)(uint, InferenceType, uint)) &
              Graph::infer_mean,
          "infer the posterior mean of the queried nodes",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401)
      .def(
          "infer_mean",
          (std::vector<std::vector<double>> &
           (Graph::*)(uint, InferenceType, uint, uint, InferConfig)) &
              Graph::infer_mean,
          "infer the posterior mean of the queried nodes using multiple chains",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401,
          py::arg("n_chains") = 4,
          py::arg("infer_config") = InferConfig())
      .def(
          "infer",
          (std::vector<std::vector<NodeValue>> &
           (Graph::*)(uint, InferenceType, uint)) &
              Graph::infer,
          "infer the empirical distribution of the queried nodes",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401)
      .def(
          "infer",
          (std::vector<std::vector<std::vector<NodeValue>>> &
           (Graph::*)(uint, InferenceType, uint, uint, InferConfig)) &
              Graph::infer,
          "infer the empirical distribution of the queried nodes using multiple chains",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401,
          py::arg("n_chains") = 4,
          py::arg("infer_config") = InferConfig())
      .def(
          "variational",
          &Graph::variational,
          "infer the empirical distribution of the queried nodes",
          py::arg("num_iters"),
          py::arg("steps_per_iter"),
          py::arg("seed") = 5123401,
          py::arg("elbo_samples") = 0)
      .def(
          "customize_transformation",
          &Graph::customize_transformation,
          "customize transformation",
          py::arg("transform_type"),
          py::arg("node_ids"))
      .def(
          "get_elbo",
          &Graph::get_elbo,
          "get the evidence lower bound (ELBO) of the last variational call")
      .def(
          "get_log_prob",
          &Graph::get_log_prob,
          "get the log probabilities of all chains")
      .def(
          "collect_performance_data",
          &Graph::collect_performance_data,
          "collect performance data",
          py::arg("b"))
      .def(
          "performance_report",
          &Graph::performance_report,
          "performance report");

  py::class_<NUTS>(module, "NUTS")
      .def(py::init<Graph&>())
      .def(
          "infer",
          &NUTS::infer,
          "infer",
          py::arg("num_samples"),
          py::arg("seed"),
          py::arg("num_warmup_samples") = 0,
          py::arg("save_warmup") = false,
          py::arg("init_type") = InitType::RANDOM);

  py::class_<HMC>(module, "HMC")
      .def(py::init<Graph&, double, double>())
      .def(
          "infer",
          &HMC::infer,
          "infer",
          py::arg("num_samples"),
          py::arg("seed"),
          py::arg("num_warmup_samples") = 0,
          py::arg("save_warmup") = false,
          py::arg("init_type") = InitType::RANDOM);
}

} // namespace graph
} // namespace beanmachine
