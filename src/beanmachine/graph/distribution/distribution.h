/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

  virtual double log_prob(const graph::NodeValue& value) const = 0;

  // The base class declares a method log_prob() that we want to preserve.
  // However, this class declared log_prob(const NodeValue&) which hides it.
  // For this reason, we must use the following using directive which
  // preserves the base class method as available.
  using graph::Node::log_prob;

  virtual void log_prob_iid(
      const graph::NodeValue& /* value */,
      Eigen::MatrixXd& /* log_probs */) const {}

  // Computes the first and second gradients of the log probability
  // with respect to given value and *adds* them to the
  // passed-by-reference parameters grad1 and grad2.
  //
  // Note that the analogy with gradient_log_prob_*param*
  // is not quite precise, because that method does *not*
  // compute the gradient with respect to the parameters,
  // but *through* them.
  //
  // See gradient_log_prob_param.
  virtual void gradient_log_prob_value(
      const graph::NodeValue& value,
      double& grad1,
      double& grad2) const = 0;

  // Computes the first and second gradients of the log probability
  // and *adds* them to the passed-by-reference grad1 and grad1 parameters.
  // Note that, similarly to Node::compute_gradient,
  // the gradient is with respect to some unspecified variable,
  // *not* with respect to the parameters of the distribution.
  // Instead, the method uses the parameters's grad1 and grad2 fields,
  // so the resulting gradient is with respect to the same variable
  // those fields are with respect to.
  //
  // Note: First order chain rule:
  // d/dx[f(g(x))]
  //     = f’(g(x)) g’(x)
  // Second order chain rule:
  // d^2/dx^2[f(g(x))]
  //     = d/dx[f’(g(x)) g’(x)] // first order chain rule
  //     = d/dx[f’(g(x))] g’(x) // product rule
  //       + d/dx[g’(x)] f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) // first order chain rule
  //       + g’’(x) f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) + g’’(x) f’(g(x))
  // Here f is log(PDF), g is the value of the parameter to the function
  // being differentiated, and g' and g'' are the incoming gradients of
  // the value with respect to the distant variable x.
  //
  // If, for some parameter p, you compute
  //   d1 = d/dp log(PDF)
  //   d2 = d^2/dp^2 log(PDF)
  // And p has incoming gradients p->grad1 and p->grad2, then this
  // code should apply the chain rule by doing
  //   grad1 += d1 * p->grad1;
  //   grad2 += d2 * p->grad1 * p->grad1 + p->grad2 * d1;
  //
  // Note that, *unlike* Node::compute_gradients, the result is stored
  // in parameters passed by reference (compute_gradients stores
  // them in the node's grad1 and grad2 fields).
  // The main reason for this is that the log prob is not itself
  // a node in the graph. Unifying these two methods is one of
  // the goals of the refactoring of the autograd component of BMG
  // planned as of May 2022.
  //
  // See gradient_log_prob_value.
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
  virtual Eigen::MatrixXd _matrix_sampler(std::mt19937& /* gen */) const {
    throw std::runtime_error(
        "_matrix_sampler has not been implemented for this distribution.");
  }
};

} // namespace distribution
} // namespace beanmachine
