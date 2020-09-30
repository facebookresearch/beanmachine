// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <string>
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace distribution {

class Distribution : public graph::Node {
 public:
  static std::unique_ptr<Distribution> new_distribution(
      graph::DistributionType dist_type,
      graph::ValueType sample_type,
      const std::vector<graph::Node*>& in_nodes);

  Distribution(graph::DistributionType dist_type, graph::AtomicType sample_type)
      : graph::Node(graph::NodeType::DISTRIBUTION),
        dist_type(dist_type),
        sample_type(sample_type) {}
  Distribution(graph::DistributionType dist_type, graph::ValueType sample_type)
      : graph::Node(graph::NodeType::DISTRIBUTION),
        dist_type(dist_type),
        sample_type(sample_type) {}
  graph::AtomicValue sample(std::mt19937& gen) const;
  void sample(std::mt19937& gen, graph::AtomicValue& sample_value) const;
  void eval(std::mt19937& /* */) override {
    throw std::runtime_error(
        "internal error: eval() is not implemented for distribution");
  }
  // tell the compiler that we want the base class log_prob method
  // as well as the new one in this class
  using graph::Node::log_prob;
  virtual double log_prob(const graph::AtomicValue& value) const = 0;
  // these function add the gradients to the passed in gradients
  virtual void gradient_log_prob_value(
      const graph::AtomicValue& value,
      double& grad1,
      double& grad2) const = 0;
  virtual void gradient_log_prob_value(
      const graph::AtomicValue& /* value */,
      Eigen::MatrixXd& /* grad1 */,
      Eigen::MatrixXd& /* grad2_diag */) const {
    throw std::runtime_error(
        "gradient_log_prob_value has not been implemented for this distribution.");
  }
  virtual void gradient_log_prob_param(
      const graph::AtomicValue& value,
      double& grad1,
      double& grad2) const = 0;
  virtual void gradient_log_prob_param(
      const graph::AtomicValue& /* value */,
      Eigen::MatrixXd& /* grad1 */,
      Eigen::MatrixXd& /* grad2_diag */) const {
    throw std::runtime_error(
        "gradient_log_prob_param has not been implemented for this distribution.");
  }
  graph::DistributionType dist_type;
  graph::ValueType sample_type;

  virtual double _double_sampler(std::mt19937& /* gen */) const {
    throw std::runtime_error(
        "_double_sampler has not been implemented for this distribution.");
  }
  virtual bool _bool_sampler(std::mt19937& /* gen */) const {
    throw std::runtime_error(
        "_bool_sampler has not been implemented for this distribution.");
  }
  virtual graph::natural_t _natural_sampler(std::mt19937& /* gen */) const {
    throw std::runtime_error(
        "_natural_sampler has not been implemented for this distribution.");
  }
};

} // namespace distribution
} // namespace beanmachine
