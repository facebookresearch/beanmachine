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
  graph::NodeValue sample(std::mt19937& gen) const;
  void sample(std::mt19937& gen, graph::NodeValue& sample_value) const;
  void eval(std::mt19937& /* */) override {
    throw std::runtime_error(
        "internal error: eval() is not implemented for distribution");
  }
  // tell the compiler that we want the base class log_prob method
  // as well as the new one in this class
  using graph::Node::log_prob;
  virtual double log_prob(const graph::NodeValue& value) const = 0;
  virtual void log_prob_iid(
      const graph::NodeValue& value,
      Eigen::MatrixXd& log_probs) const {}
  // these function add the gradients to the passed in gradients
  virtual void gradient_log_prob_value(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const = 0;
  virtual void gradient_log_prob_param(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const = 0;

  /*
  In backward gradient propagation, increments the back_grad by the gradient of
  the log prob of the distribution w.r.t. the sampled value.
  :param value: value of the child Sample operator, a single draw from the
  distribution
  :param back_grad: back_grad1 of the child Sample operator, to be incremented
  :param adjunct: a multiplier that represents the gradient of the target
  function w.r.t the log prob of this distribution. It uses the default value
  1.0 if the direct child is a StochasticOperator, but requires input if the
  direct child is a mixture distribution.
  */
  virtual void backward_value(
      const graph::NodeValue& /* value */,
      graph::DoubleMatrix& /* back_grad */,
      double /* adjunct */ = 1.0) const {}
  /*
  Similar to backward_value, except that it is used when the child operator is
  IId_Sample
  */
  virtual void backward_value_iid(
      const graph::NodeValue& /* value */,
      graph::DoubleMatrix& /* back_grad */) const {}
  virtual void backward_value_iid(
      const graph::NodeValue& /* value */,
      graph::DoubleMatrix& /* back_grad */,
      Eigen::MatrixXd& /* adjunct */) const {}
  /*
  In backward gradient propagation, increments the back_grad1 of each parent
  node w.r.t. the log prob of the distribution, evaluated at the given value.
  :param value: value of the child Sample operator, a single draw from the
  distribution
  :param adjunct: a multiplier that represents the gradient of the
  target function w.r.t the log prob of this distribution. It uses the default
  value 1.0 if the direct child is a StochasticOperator, but requires input if
  the direct child is a mixture distribution.
  */
  virtual void backward_param(
      const graph::NodeValue& /* value */,
      double /* adjunct */ = 1.0) const {}
  /*
  Similar to backward_param, except that it is used when the child operator is
  IId_Sample
  */
  virtual void backward_param_iid(const graph::NodeValue& /* value */) const {}
  virtual void backward_param_iid(
      const graph::NodeValue& /* value */,
      Eigen::MatrixXd& /* adjunct */) const {}
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
