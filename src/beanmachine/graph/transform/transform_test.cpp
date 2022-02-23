/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

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
  torch::Tensor pos2_obs(2, 1);
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
  y->_value = 0.0;
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
  torch::Tensor xobs(2, 1);
  xobs << 0.5, 1.5;
  g2.observe(x2, xobs);

  // To verify the results with pyTorch:
  // log_a = torch.tensor(np.log(10.0), requires_grad=True)
  // log_b = torch.tensor(np.log(1.2), requires_grad=True)
  // log_x = torch.tensor(np.log([2.5, 0.5, 1.5]), requires_grad=True)
  // log_p = torch.distributions.Gamma(
  //   log_a.exp(), log_b.exp()).log_prob(log_x.exp()).sum()
  // log_q = log_p + log_a + log_b + log_x.sum()
  // torch.autograd.grad(log_q, log_x)[0]
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 4);
  EXPECT_NEAR(back_grad[0]->_double, -54.7968, 0.001); // log rate1
  EXPECT_NEAR(back_grad[1]->_double, 25.6, 0.001); // log shape1
  EXPECT_NEAR(back_grad[2]->_double, 7.0, 0.001); // log x1
  EXPECT_NEAR(back_grad[3]->_matrix.coeff(0), 9.4, 0.001); // log x2
  EXPECT_NEAR(back_grad[3]->_matrix.coeff(1), 8.2, 0.001);
  EXPECT_NEAR(g2.full_log_prob(), -29.5648, 1e-3);
}

TEST(test_transform, unconstrained_type) {
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
