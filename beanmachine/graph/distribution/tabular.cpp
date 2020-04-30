// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>
#include <random>

#include "beanmachine/graph/distribution/tabular.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

Tabular::Tabular(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::TABULAR, sample_type) {
  // check the sample datatype
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument("Tabular supports only boolean valued samples");
  }
  // extract the matrix from the first parent
  if (in_nodes.size() < 1 or
      in_nodes[0]->node_type != graph::NodeType::CONSTANT or
      in_nodes[0]->value.type != graph::AtomicType::TENSOR) {
    throw std::invalid_argument(
        "Tabular distribution first arg must be tensor");
  }
  const torch::Tensor matrix = in_nodes[0]->value._tensor;
  if (not util::is_real_tensor(matrix)) {
    throw std::invalid_argument(
        "Tabular distribution's tensor argument should have type float");
  }
  // check the dimensions of the matrix
  const torch::IntArrayRef sizes = matrix.sizes();
  // we need one dimension for each parent excluding the first tensor parent
  // and one for the output
  if (sizes.size() != in_nodes.size()) {
    throw std::invalid_argument(
        "Tabular distribution's tensor argument expected " +
        std::to_string(in_nodes.size()) + " dims got " +
        std::to_string(sizes.size()));
  }
  // since only boolean sample types are currently supported the last dimension
  // of the matrix must be 2
  if (sizes[sizes.size() - 1] != 2) {
    throw std::invalid_argument(
        "Tabular distribution's tensor should have last dimension size 2");
  }
  // go through each of the parents other than the tensor and verify its type
  for (uint paridx = 1; paridx < in_nodes.size(); paridx++) {
    const graph::Node* parent = in_nodes[paridx];
    if (parent->value.type != graph::AtomicType::BOOLEAN) {
      throw std::invalid_argument(
          "Tabular distribution only supports boolean parents currently");
    }
  }
  // check that the matrix defines a probability distribution in the last dim
  if (matrix.lt(0).any().item<uint8_t>()) {
    throw std::invalid_argument("Tabular distribution tensor must be positive");
  }
  if (matrix.sum(-1).sub(1).abs().gt(1e-6).any().item<uint8_t>()) {
    throw std::invalid_argument(
        "Tabular distribution tensor last dim must add to 1");
  }
}

double Tabular::get_probability() const {
  std::vector<torch::Tensor> parents;
  for (uint i = 1; i < in_nodes.size(); i++) {
    const auto& parenti = in_nodes[i]->value;
    if (parenti.type != graph::AtomicType::BOOLEAN) {
      throw std::runtime_error(
          "Tabular distribution at node_id " + std::to_string(index) +
          " expects boolean parents");
    }
    parents.push_back(
        torch::scalar_tensor((int64_t)parenti._bool, torch::kLong));
  }
  parents.push_back(torch::scalar_tensor((int64_t)1, torch::kLong));
  assert(in_nodes[0]->value.type == graph::AtomicType::TENSOR);
  double prob = in_nodes[0]->value._tensor.index(parents).item<double>();
  if (prob < 0 or prob > 1) {
    throw std::runtime_error(
        "unexpected probability " + std::to_string(prob) +
        " in Tabular node_id " + std::to_string(index));
  }
  return prob;
}

graph::AtomicValue Tabular::sample(std::mt19937& gen) const {
  double prob_true = get_probability();
  std::bernoulli_distribution distrib(prob_true);
  return graph::AtomicValue((bool)distrib(gen));
}

double Tabular::log_prob(const graph::AtomicValue& value) const {
  double prob_true = get_probability();
  if (value.type != graph::AtomicType::BOOLEAN) {
    throw std::runtime_error(
        "expecting boolean value in child of Tabular node_id " +
        std::to_string(index) + " got type " +
        std::to_string(static_cast<int>(value.type)));
  }
  return value._bool ? std::log(prob_true) : std::log(1 - prob_true);
}

void Tabular::gradient_log_prob_value(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  throw std::runtime_error(
      "gradient_log_prob_value not implemented for Tabular");
}

void Tabular::gradient_log_prob_param(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  throw std::runtime_error(
      "gradient_log_prob_param not implemented for Tabular");
}

} // namespace distribution
} // namespace beanmachine
