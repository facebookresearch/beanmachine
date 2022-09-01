/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <beanmachine/graph/operator/operator.h>
#include "beanmachine/graph/distribution/flat.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/transform/transform.h"

using namespace beanmachine;
using namespace beanmachine::graph;

TEST(test_transform, log) {
  std::mt19937 generator(1234);
  Graph g1;
  NodeValue *x, *y;
  auto size = g1.add_constant((natural_t)2);
  auto flat_real = g1.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto real1 =
      g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  // negative test: Log only applies to POS_REAL
  EXPECT_THROW(
      g1.customize_transformation(TransformType::LOG, std::vector<uint>{real1}),
      std::invalid_argument);
  // test transform and inverse transform
  auto flat_pos = g1.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto pos1 =
      g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto pos2 = g1.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{flat_pos, size});
  g1.customize_transformation(
      TransformType::LOG, std::vector<uint>{pos1, pos2});
  g1.observe(pos1, 2.5);
  Eigen::MatrixXd pos2_obs(2, 1);
  pos2_obs << 0.5, 1.5;
  g1.observe(pos2, pos2_obs);
  // scalar transform
  auto n1 = static_cast<oper::StochasticOperator*>(
      g1.check_node(pos1, NodeType::OPERATOR));
  y = n1->get_unconstrained_value(false);
  EXPECT_NEAR(y->_double, 0, 0.001);
  x = n1->get_original_value(false);
  EXPECT_NEAR(x->_double, 2.5, 0.001);
  y = n1->get_unconstrained_value(true);
  EXPECT_NEAR(y->_double, std::log(2.5), 0.001);
  y->_double = 0.0;
  x = n1->get_original_value(true);
  EXPECT_NEAR(x->_double, 1.0, 0.001);
  // vector transform
  auto n2 = static_cast<oper::StochasticOperator*>(
      g1.check_node(pos2, NodeType::OPERATOR));
  y = n2->get_unconstrained_value(false);
  EXPECT_NEAR(y->_matrix.squaredNorm(), 0, 0.001);
  x = n2->get_original_value(false);
  EXPECT_NEAR(x->_matrix.coeff(0), 0.5, 0.001);
  EXPECT_NEAR(x->_matrix.coeff(1), 1.5, 0.001);
  y = n2->get_unconstrained_value(true);
  EXPECT_NEAR(y->_matrix.coeff(0), std::log(0.5), 0.001);
  EXPECT_NEAR(y->_matrix.coeff(1), std::log(1.5), 0.001);
  y->_matrix.setZero();
  x = n2->get_original_value(true);
  EXPECT_NEAR(x->_matrix.coeff(0), 1.0, 0.001);
  EXPECT_NEAR(x->_matrix.coeff(1), 1.0, 0.001);

  // test log_abs_jacobian_determinant and unconstrained_gradient
  Graph g2;
  size = g2.add_constant((natural_t)2);
  flat_pos = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto rate1 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto shape1 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto dist = g2.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{rate1, shape1});
  auto x1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto x2 =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.customize_transformation(
      TransformType::LOG, std::vector<uint>{rate1, shape1, x1, x2});
  g2.observe(rate1, 10.0);
  g2.observe(shape1, 1.2);
  g2.observe(x1, 2.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, 1.5;
  g2.observe(x2, xobs);

  /*
  # To verify the results with pyTorch:
from torch import tensor
from torch import log
from torch.distributions import Gamma
from torch.autograd import grad

rate1 = tensor(10.0, requires_grad=True)
log_rate1 = log(rate1)
shape1 = tensor(1.2, requires_grad=True)
log_shape1 = log(shape1)
x = tensor([2.5, 0.5, 1.5], requires_grad=True)
log_x = tensor(log(x), requires_grad=True)
gamma = Gamma(
    log_rate1.exp(),
    log_shape1.exp())
log_p = gamma.log_prob(log_x.exp()).sum()
log_q = log_p + log_rate1 + log_shape1 + log_x.sum()
print("rate1", grad(log_q, log_rate1, retain_graph=True))
print("shape1", grad(log_q, log_shape1, retain_graph=True))
print("x", grad(log_q, log_x, retain_graph=True))
print("full_log_prob", log_q)
  */
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 4);
  EXPECT_NEAR((*back_grad[0]), -54.7968, 0.001); // log rate1
  EXPECT_NEAR((*back_grad[1]), 25.6, 0.001); // log shape1
  EXPECT_NEAR((*back_grad[2]), 7.0, 0.001); // log x1
  EXPECT_NEAR(back_grad[3]->coeff(0), 9.4, 0.001); // log x2
  EXPECT_NEAR(back_grad[3]->coeff(1), 8.2, 0.001);
  EXPECT_NEAR(g2.full_log_prob(), -29.5648, 1e-3);
}

namespace {
double logit(double p) {
  return std::log(p / (1 - p));
}
double expit(double x) {
  return 1 / (1 + std::exp(-x));
}
} // namespace

TEST(test_transform, sigmoid_flat) {
  std::mt19937 generator(1234);
  Graph g1;
  NodeValue *x, *y;
  auto size = g1.add_constant((natural_t)2);
  auto flat_real =
      g1.add_distribution(DistributionType::FLAT, AtomicType::REAL, {});
  auto real1 = g1.add_operator(OperatorType::SAMPLE, {flat_real});
  // negative test: Sigmoid only applies to PROBABILITY
  EXPECT_THROW(
      g1.customize_transformation(TransformType::SIGMOID, {real1}),
      std::invalid_argument);
  // test transform and inverse transform
  auto flat_pos =
      g1.add_distribution(DistributionType::FLAT, AtomicType::PROBABILITY, {});
  auto pos1 = g1.add_operator(OperatorType::SAMPLE, {flat_pos});
  auto pos2 = g1.add_operator(OperatorType::IID_SAMPLE, {flat_pos, size});
  g1.customize_transformation(TransformType::SIGMOID, {pos1, pos2});
  g1.observe(pos1, 0.2);
  Eigen::MatrixXd pos2_obs(2, 1);
  pos2_obs << 0.4, 0.5;
  g1.observe(pos2, pos2_obs);

  // scalar transform
  auto n1 = static_cast<oper::StochasticOperator*>(
      g1.check_node(pos1, NodeType::OPERATOR));
  y = n1->get_unconstrained_value(false);
  EXPECT_NEAR(y->_double, 0, 0.001);
  x = n1->get_original_value(false);
  EXPECT_NEAR(x->_double, 0.2, 0.001);
  y = n1->get_unconstrained_value(true);
  EXPECT_NEAR(y->_double, logit(0.2), 0.001);
  y->_double = 0.0;
  x = n1->get_original_value(true);
  EXPECT_NEAR(x->_double, expit(0.0), 0.001);

  // vector transform
  auto n2 = static_cast<oper::StochasticOperator*>(
      g1.check_node(pos2, NodeType::OPERATOR));
  y = n2->get_unconstrained_value(false);
  EXPECT_NEAR(y->_matrix.squaredNorm(), 0, 0.001);
  x = n2->get_original_value(false);
  EXPECT_NEAR(x->_matrix.coeff(0), 0.4, 0.001);
  EXPECT_NEAR(x->_matrix.coeff(1), 0.5, 0.001);
  y = n2->get_unconstrained_value(true);
  EXPECT_NEAR(y->_matrix.coeff(0), logit(0.4), 0.001);
  EXPECT_NEAR(y->_matrix.coeff(1), logit(0.5), 0.001);
  y->_matrix.setZero();
  x = n2->get_original_value(true);
  EXPECT_NEAR(x->_matrix.coeff(0), expit(0.0), 0.001);
  EXPECT_NEAR(x->_matrix.coeff(1), expit(0.0), 0.001);

  Graph g2;
  size = g2.add_constant((natural_t)2);
  auto dist =
      g2.add_distribution(DistributionType::FLAT, AtomicType::PROBABILITY, {});
  auto x1 = g2.add_operator(OperatorType::SAMPLE, {dist});
  auto x2 = g2.add_operator(OperatorType::IID_SAMPLE, {dist, size});
  g2.customize_transformation(TransformType::SIGMOID, {x1, x2});
  g2.observe(x1, 0.2);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.4, 0.5;
  g2.observe(x2, xobs);
}

TEST(test_transform, sigmoid_beta) {
  std::mt19937 generator(1234);
  Graph g1;
  NodeValue *x, *y;
  auto size = g1.add_constant((natural_t)2);
  auto two = g1.add_constant_pos_real(2.0);
  auto beta = g1.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, {two, two});
  auto pos1 = g1.add_operator(OperatorType::SAMPLE, {beta});
  auto pos2 = g1.add_operator(OperatorType::IID_SAMPLE, {beta, size});
  g1.customize_transformation(TransformType::SIGMOID, {pos1, pos2});
  g1.observe(pos1, 0.2);
  Eigen::MatrixXd pos2_obs(2, 1);
  pos2_obs << 0.4, 0.5;
  g1.observe(pos2, pos2_obs);
  // scalar transform
  auto n1 = static_cast<oper::StochasticOperator*>(
      g1.check_node(pos1, NodeType::OPERATOR));
  y = n1->get_unconstrained_value(false);
  EXPECT_NEAR(y->_double, 0, 0.001);
  x = n1->get_original_value(false);
  EXPECT_NEAR(x->_double, 0.2, 0.001);
  y = n1->get_unconstrained_value(true);
  EXPECT_NEAR(y->_double, std::log(0.25), 0.001);
  y->_double = 0.0;
  x = n1->get_original_value(true);
  EXPECT_NEAR(x->_double, expit(0.0), 0.001);
  // vector transform
  auto n2 = static_cast<oper::StochasticOperator*>(
      g1.check_node(pos2, NodeType::OPERATOR));
  y = n2->get_unconstrained_value(false);
  EXPECT_NEAR(y->_matrix.squaredNorm(), 0, 0.001);
  x = n2->get_original_value(false);
  EXPECT_NEAR(x->_matrix.coeff(0), 0.4, 0.001);
  EXPECT_NEAR(x->_matrix.coeff(1), 0.5, 0.001);
  y = n2->get_unconstrained_value(true);
  EXPECT_NEAR(y->_matrix.coeff(0), std::log(2.0 / 3), 0.001);
  EXPECT_NEAR(y->_matrix.coeff(1), 0, 0.001);
  y->_matrix.setZero();
  x = n2->get_original_value(true);
  EXPECT_NEAR(x->_matrix.coeff(0), expit(0), 0.001);
  EXPECT_NEAR(x->_matrix.coeff(1), expit(0), 0.001);
}

TEST(test_transform, sigmoid_beta_2) {
  // test log_abs_jacobian_determinant and unconstrained_gradient
  Graph g2;
  auto dist = g2.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      {g2.add_constant_pos_real(0.25), g2.add_constant_pos_real(0.75)});

  auto x1 = g2.add_operator(OperatorType::SAMPLE, {dist});
  g2.customize_transformation(TransformType::SIGMOID, {x1});
  g2.observe(x1, 0.5);

  auto x2 = g2.add_operator(
      OperatorType::IID_SAMPLE, {dist, g2.add_constant((natural_t)2)});
  g2.customize_transformation(TransformType::SIGMOID, {x2});
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.2, 0.3;
  g2.observe(x2, xobs);

  /*
  # To verify the results with pyTorch:
from torch import tensor
from torch import logit
from torch.special import expit
from torch.autograd import grad
from torch.distributions import Beta
from torch.distributions import TransformedDistribution
from torch.distributions import SigmoidTransform

a = tensor(0.25, requires_grad=True)
b = tensor(0.75, requires_grad=True)
x = tensor([0.5, 0.2, 0.3], requires_grad=True)
logit_x = logit(x)

beta_dist = Beta(a, b)
log_p = beta_dist.log_prob(expit(logit_x)).sum()

transformed_dist = TransformedDistribution(
    beta_dist, [SigmoidTransform().inv])
full_log_prob = transformed_dist.log_prob(logit_x).sum()

print("xgrad", grad(full_log_prob, logit_x, retain_graph=True))
print("full_log_prob", full_log_prob)
  */
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 2);
  // The indices below are of the random variables in the model, not node
  // indices.
  EXPECT_NEAR(g2.full_log_prob(), -6.3053, 1e-3);
  EXPECT_NEAR((*back_grad[0]), -0.2500, 1e-3); // x1
  EXPECT_NEAR(back_grad[1]->coeff(0), 0.0500, 1e-3); // x2[0]
  EXPECT_NEAR(back_grad[1]->coeff(1), -0.0500, 1e-3); // x2[1]
}

TEST(test_transform, log_unconstrained_type) {
  Graph g1;
  auto size = g1.add_constant((natural_t)2);

  // test transform types
  auto flat_pos = g1.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto sample =
      g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto iid_sample = g1.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{flat_pos, size});
  g1.customize_transformation(
      TransformType::LOG, std::vector<uint>{sample, iid_sample});

  auto n1 = static_cast<oper::StochasticOperator*>(
      g1.check_node(sample, NodeType::OPERATOR));
  EXPECT_EQ(n1->value.type.atomic_type, AtomicType::POS_REAL);
  EXPECT_EQ(n1->unconstrained_value.type.atomic_type, AtomicType::REAL);

  auto n2 = static_cast<oper::StochasticOperator*>(
      g1.check_node(iid_sample, NodeType::OPERATOR));
  EXPECT_EQ(n2->value.type.atomic_type, AtomicType::POS_REAL);
  // check type is unknown before calling "get_unconstrained_value"
  EXPECT_EQ(n2->unconstrained_value.type.atomic_type, AtomicType::UNKNOWN);
  // check that the type is initialized properly
  EXPECT_EQ(
      n2->get_unconstrained_value(true)->type.atomic_type, AtomicType::REAL);
}

TEST(test_transform, sigmoid_unconstrained_type) {
  Graph g1;
  auto size = g1.add_constant((natural_t)2);

  // test transform types
  auto flat_pos =
      g1.add_distribution(DistributionType::FLAT, AtomicType::PROBABILITY, {});
  auto sample = g1.add_operator(OperatorType::SAMPLE, {flat_pos});
  auto iid_sample = g1.add_operator(OperatorType::IID_SAMPLE, {flat_pos, size});
  g1.customize_transformation(TransformType::SIGMOID, {sample, iid_sample});

  auto n1 = static_cast<oper::StochasticOperator*>(
      g1.check_node(sample, NodeType::OPERATOR));
  EXPECT_EQ(n1->value.type.atomic_type, AtomicType::PROBABILITY);
  EXPECT_EQ(n1->unconstrained_value.type.atomic_type, AtomicType::REAL);

  auto n2 = static_cast<oper::StochasticOperator*>(
      g1.check_node(iid_sample, NodeType::OPERATOR));
  EXPECT_EQ(n2->value.type.atomic_type, AtomicType::PROBABILITY);
  // check type is unknown before calling "get_unconstrained_value"
  EXPECT_EQ(n2->unconstrained_value.type.atomic_type, AtomicType::UNKNOWN);
  // check that the type is initialized properly
  EXPECT_EQ(
      n2->get_unconstrained_value(true)->type.atomic_type, AtomicType::REAL);
}
