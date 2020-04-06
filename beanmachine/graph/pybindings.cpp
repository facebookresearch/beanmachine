// Copyright (c) Facebook, Inc. and its affiliates.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define TORCH_API_INCLUDE_EXTENSION_H 1
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

namespace py = pybind11;

PYBIND11_MODULE(graph, module) {
  module.doc() = "module for python bindings to the graph API";

  py::enum_<AtomicType>(module, "AtomicType")
      .value("BOOLEAN", AtomicType::BOOLEAN)
      .value("REAL", AtomicType::REAL)
      .value("PROBABILITY", AtomicType::PROBABILITY)
      .value("NATURAL", AtomicType::NATURAL)
      .value("TENSOR", AtomicType::TENSOR);

  py::class_<AtomicValue>(module, "AtomicValue")
      .def(py::init<bool>())
      .def(py::init<double>())
      .def(py::init<graph::natural_t>())
      .def(py::init<torch::Tensor>())
      .def_readonly("type", &AtomicValue::type)
      .def_readonly("bool", &AtomicValue::_bool)
      .def_readonly("real", &AtomicValue::_double)
      .def_readonly("probability", &AtomicValue::_double)
      .def_readonly("natural", &AtomicValue::_natural)
      .def_readonly("tensor", &AtomicValue::_tensor);

  py::enum_<OperatorType>(module, "OperatorType")
      .value("SAMPLE", OperatorType::SAMPLE)
      .value("TO_REAL", OperatorType::TO_REAL)
      .value("TO_POS_REAL", OperatorType::TO_POS_REAL)
      .value("TO_TENSOR", OperatorType::TO_TENSOR)
      .value("COMPLEMENT", OperatorType::COMPLEMENT)
      .value("NEGATE", OperatorType::NEGATE)
      .value("EXP", OperatorType::EXP)
      .value("EXPM1", OperatorType::EXPM1)
      .value("MULTIPLY", OperatorType::MULTIPLY)
      .value("ADD", OperatorType::ADD);

  py::enum_<DistributionType>(module, "DistributionType")
      .value("TABULAR", DistributionType::TABULAR)
      .value("BERNOULLI", DistributionType::BERNOULLI)
      .value("BERNOULLI_NOISY_OR", DistributionType::BERNOULLI_NOISY_OR)
      .value("BETA", DistributionType::BETA)
      .value("BINOMIAL", DistributionType::BINOMIAL)
      .value("FLAT", DistributionType::FLAT)
      .value("NORMAL", DistributionType::NORMAL);

  py::enum_<NodeType>(module, "NodeType")
      .value("CONSTANT", NodeType::CONSTANT)
      .value("DISTRIBUTION", NodeType::DISTRIBUTION)
      .value("OPERATOR", NodeType::OPERATOR);

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
          (uint(Graph::*)(torch::Tensor)) & Graph::add_constant,
          "add a Node with a constant tensor value",
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
          "add_distribution",
          &Graph::add_distribution,
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
          (void (Graph::*)(uint, torch::Tensor)) & Graph::observe,
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
          &Graph::infer_mean,
          "infer the posterior mean of the queried nodes",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401)
      .def(
          "infer",
          &Graph::infer,
          "infer the empirical distribution of the queried nodes",
          py::arg("num_samples"),
          py::arg("algorithm") = InferenceType::GIBBS,
          py::arg("seed") = 5123401)
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
