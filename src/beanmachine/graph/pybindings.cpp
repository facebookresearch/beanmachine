// Copyright (c) Facebook, Inc. and its affiliates.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "beanmachine/graph/pybindings.h"


namespace beanmachine {
namespace graph {

namespace py = pybind11;

PYBIND11_MODULE(graph, module) {
  module.doc() = "module for python bindings to the graph API";

  py::enum_<VariableType>(module, "VariableType")
      .value("SCALAR", VariableType::SCALAR)
      .value("BROADCAST_MATRIX", VariableType::BROADCAST_MATRIX)
      .value("COL_SIMPLEX_MATRIX", VariableType::COL_SIMPLEX_MATRIX);

  py::enum_<AtomicType>(module, "AtomicType")
      .value("BOOLEAN", AtomicType::BOOLEAN)
      .value("PROBABILITY", AtomicType::PROBABILITY)
      .value("REAL", AtomicType::REAL)
      .value("POS_REAL", AtomicType::POS_REAL)
      .value("NATURAL", AtomicType::NATURAL);

  py::class_<ValueType>(module, "ValueType")
      .def(py::init<VariableType, AtomicType, uint, uint>())
      .def(
          "to_string",
          &ValueType::to_string,
          "string representation of the type");

  py::class_<AtomicValue>(module, "AtomicValue")
      .def(py::init<bool>())
      .def(py::init<double>())
      .def(py::init<graph::natural_t>())
      .def(py::init<Eigen::MatrixXb&>())
      .def(py::init<Eigen::MatrixXd&>());

  py::enum_<OperatorType>(module, "OperatorType")
      .value("SAMPLE", OperatorType::SAMPLE)
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
      .value("LOGSUMEXP", OperatorType::LOGSUMEXP)
      .value("IF_THEN_ELSE", OperatorType::IF_THEN_ELSE)
      .value("LOG", OperatorType::LOG)
      .value("POW", OperatorType::POW)
      .value("NEGATIVE_LOG", OperatorType::NEGATIVE_LOG)
      .value("MATRIX_MULTIPLY", OperatorType::MATRIX_MULTIPLY);

  py::enum_<DistributionType>(module, "DistributionType")
      .value("TABULAR", DistributionType::TABULAR)
      .value("BERNOULLI", DistributionType::BERNOULLI)
      .value("BERNOULLI_NOISY_OR", DistributionType::BERNOULLI_NOISY_OR)
      .value("BETA", DistributionType::BETA)
      .value("BINOMIAL", DistributionType::BINOMIAL)
      .value("FLAT", DistributionType::FLAT)
      .value("NORMAL", DistributionType::NORMAL)
      .value("HALF_CAUCHY", DistributionType::HALF_CAUCHY)
      .value("STUDENT_T", DistributionType::STUDENT_T)
      .value("BERNOULLI_LOGIT", DistributionType::BERNOULLI_LOGIT)
      .value("GAMMA", DistributionType::GAMMA)
      .value("BIMIXTURE", DistributionType::BIMIXTURE);

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

  py::class_<Graph>(module, "Graph")
      .def(py::init())
      .def("to_string", &Graph::to_string, "string representation of the graph")
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
          (uint(Graph::*)(AtomicValue)) & Graph::add_constant,
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
          "add_constant_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) & Graph::add_constant_matrix,
          "add a Node with a constant real-valued matrix",
          py::arg("value"))
      .def(
          "add_constant_matrix",
          (uint(Graph::*)(Eigen::MatrixXb&)) & Graph::add_constant_matrix,
          "add a Node with a constant boolean-valued matrix",
          py::arg("value"))
      .def(
          "add_constant_matrix",
          (uint(Graph::*)(Eigen::MatrixXn&)) & Graph::add_constant_matrix,
          "add a Node with a constant natural_t-valued matrix",
          py::arg("value"))
      .def(
          "add_constant_pos_matrix",
          (uint(Graph::*)(Eigen::MatrixXd&)) & Graph::add_constant_pos_matrix,
          "add a Node with a constant element-wise positive valued matrix",
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
          (void (Graph::*)(uint, AtomicValue)) & Graph::observe,
          "observe a node",
          py::arg("node_id"),
          py::arg("val"))
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
           (Graph::*)(uint, InferenceType, uint, uint)) &
              Graph::infer_mean,
          "infer the posterior mean of the queried nodes using multiple chains",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401,
          py::arg("n_chains"))
      .def(
          "infer",
          (std::vector<std::vector<AtomicValue>> &
           (Graph::*)(uint, InferenceType, uint)) &
              Graph::infer,
          "infer the empirical distribution of the queried nodes",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401)
      .def(
          "infer",
          (std::vector<std::vector<std::vector<AtomicValue>>> &
           (Graph::*)(uint, InferenceType, uint, uint)) &
              Graph::infer,
          "infer the empirical distribution of the queried nodes using multiple chains",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401,
          py::arg("n_chains"))
      .def(
          "variational",
          &Graph::variational,
          "infer the empirical distribution of the queried nodes",
          py::arg("num_iters"),
          py::arg("steps_per_iter"),
          py::arg("seed") = 5123401,
          py::arg("elbo_samples") = 0)
      .def(
          "get_elbo",
          &Graph::get_elbo,
          "get the evidence lower bound (ELBO) of the last variational call");
}

} // namespace graph
} // namespace beanmachine
