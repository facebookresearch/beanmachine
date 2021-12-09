/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <tuple>

#include <gtest/gtest.h>

#include "beanmachine/graph/global/global_state.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, global_state_no_transform) {
  // test GlobalState: setting + getting global values,
  // incrementing global values, and calculating gradients
  Graph g = Graph();

  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant((natural_t)2);
  uint three = g.add_constant(3.0);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});
  uint sample3 = g.add_operator(OperatorType::MULTIPLY, {sample, three});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, one});
  uint sample_sample =
      g.add_operator(OperatorType::IID_SAMPLE, {norm_norm_dist, two});

  g.query(sample);
  g.query(sample_sample);
  g.query(sample3);

  GlobalState state = GlobalState(g);

  Eigen::VectorXd unconstrained_values(3);
  Eigen::VectorXd get_flattened_values;

  // negative test for set_flattened_unconstrained_values
  // with invalid dimensions
  // (model has 3 values but we are passing in vector with only 2)
  Eigen::VectorXd incorrect_flattened_values(2);
  incorrect_flattened_values << 1.0, 2.0;
  EXPECT_THROW(
      state.set_flattened_unconstrained_values(incorrect_flattened_values),
      std::invalid_argument);

  // test set_flattened_unconstrained_values,
  // which sets the values of all the unobserved stochastic
  // nodes in the graph
  // and get_flattened_unconstrained_values,
  // which reads these values from the nodes in graph
  Eigen::VectorXd flattened_values(3);
  flattened_values << 1.0, 2.0, 3.0;
  state.set_flattened_unconstrained_values(flattened_values);
  state.get_flattened_unconstrained_values(get_flattened_values);
  EXPECT_EQ(get_flattened_values.size(), 3);
  EXPECT_EQ(get_flattened_values[0], flattened_values[0]);
  EXPECT_EQ(get_flattened_values[1], flattened_values[1]);
  EXPECT_EQ(get_flattened_values[2], flattened_values[2]);

  // negative test for add_to_stochastic_unconstrained_nodes
  // where the vector has an incorrect shape
  Eigen::VectorXd incorrect_addition(1);
  incorrect_addition << 1.0;
  EXPECT_THROW(
      state.add_to_stochastic_unconstrained_nodes(incorrect_addition),
      std::invalid_argument);

  // test add_to_stochastic_unconstrained_nodes
  // changes the values appropriately
  Eigen::VectorXd addition(3);
  addition << 2.0, 4.0, 6.0;
  state.add_to_stochastic_unconstrained_nodes(addition);
  state.get_flattened_unconstrained_values(get_flattened_values);
  Eigen::VectorXd incremented_values = flattened_values + addition;
  EXPECT_EQ(get_flattened_values.size(), 3);
  EXPECT_EQ(get_flattened_values[0], incremented_values[0]);
  EXPECT_EQ(get_flattened_values[1], incremented_values[1]);
  EXPECT_EQ(get_flattened_values[2], incremented_values[2]);

  // test backup_unconstrained_values and revert_unconstrained_values
  // where revert_unconstrained_values resets the values to the
  // latest point where backup_unconstrained_values was called
  state.backup_unconstrained_values();
  state.add_to_stochastic_unconstrained_nodes(addition);
  state.revert_unconstrained_values();
  state.get_flattened_unconstrained_values(get_flattened_values);
  EXPECT_EQ(get_flattened_values.size(), 3);
  EXPECT_EQ(get_flattened_values[0], incremented_values[0]);
  EXPECT_EQ(get_flattened_values[1], incremented_values[1]);
  EXPECT_EQ(get_flattened_values[2], incremented_values[2]);

  // test back_grad is being propagated correctly
  // after unconstrained values are set
  unconstrained_values << 0.5, 0.8, 0.4;
  state.set_flattened_unconstrained_values(unconstrained_values);
  std::vector<DoubleMatrix*> back_grad;
  g.test_grad(back_grad);
  /*
  norm_dist = dist.Normal(0, 1)
  sample = tensor(0.5, requires_grad=True)
  norm2_dist = dist.Normal(sample, 1)
  sample2 = tensor([0.8, 0.4], requires_grad=True)
  log_prob = norm_dist.log_prob(sample) + norm2_dist.log_prob(sample2).sum()
  log_prob <- -2.9318
  grad(log_prob, sample, retain_graph=True) <- -0.3
  grad(log_prob, sample2) <- 0.3, 0.1
  */
  EXPECT_EQ(back_grad.size(), 2);
  EXPECT_NEAR(back_grad[0]->_double, -0.3, 1e-4);
  EXPECT_NEAR(back_grad[1]->_matrix(0), -0.3, 1e-4);
  EXPECT_NEAR(back_grad[1]->_matrix(1), 0.1, 1e-4);
  // test update_log_prob after unconstrained values are set
  state.update_log_prob();
  EXPECT_NEAR(state.get_log_prob(), -2.9318, 0.001);

  // test backup_unconstrained_grad and revert_unconstrained_grad
  // where revert_unconstrained_grad resets the back_grad to
  // previous value where backup_unconstrained_grad was called
  state.backup_unconstrained_values();
  state.backup_unconstrained_grads();
  flattened_values << 0.2, 0.3, 0.4;
  state.set_flattened_unconstrained_values(flattened_values);
  g.test_grad(back_grad);
  state.revert_unconstrained_grads();
  // gradients should be same as before, since we have reverted them
  EXPECT_NEAR(back_grad[0]->_double, -0.3, 1e-4);
  EXPECT_NEAR(back_grad[1]->_matrix(0), -0.3, 1e-4);
  EXPECT_NEAR(back_grad[1]->_matrix(1), 0.1, 1e-4);
}

TEST(testglobal, global_state_transform) {
  Graph g;
  uint one = g.add_constant_pos_real(1.0);
  uint three_nat = g.add_constant((natural_t)3);
  uint three = g.add_constant_pos_real(3.0);

  uint dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{one, three});
  uint x1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  uint x2 = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{dist, three_nat});
  g.customize_transformation(TransformType::LOG, std::vector<uint>{x1, x2});
  g.query(x1);
  g.query(x2);

  GlobalState state = GlobalState(g);

  Eigen::VectorXd unconstrained_values(4);
  unconstrained_values << -1.2, -2.0, -0.4, -0.8;
  state.set_flattened_unconstrained_values(unconstrained_values);
  /*
  unconstrained_value = tensor([-1.2, -2.0, -0.4, -0.8])
  log_transform = ExpTransform().inv
  transformed_dist = TransformedDistribution(Gamma(1, 3), log_transform)
  transformed_dist.log_prob(unconstrained_value).sum()
  */
  EXPECT_NEAR(g.full_log_prob(), -4.6741, 1e-3);

  Eigen::VectorXd increment(4);
  increment << -0.1, -0.1, -0.1, -0.1;
  /*
  unconstrained_value = tensor([-1.3, -2.1, -0.5, -0.9])
  log_transform = ExpTransform().inv
  transformed_dist = TransformedDistribution(Gamma(1, 3), log_transform)
  transformed_dist.log_prob(unconstrained_value).sum()
  */
  state.add_to_stochastic_unconstrained_nodes(increment);
  EXPECT_NEAR(g.full_log_prob(), -4.6298, 1e-3);
}

TEST(testglobal, global_state_gamma_transform_obs) {
  /*
  Testing global state with a single log-transformation of
  a single variable.

  We use the following model:
  p ~ Gamma(1, 2)    <- transform to logspace
  p2 ~ Gamma(3, p)   <- observed, value = 0.8
  */
  Graph g;
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint three = g.add_constant_pos_real(3.0);

  uint g_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{one, two});
  uint p = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{g_dist});
  g.customize_transformation(TransformType::LOG, std::vector<uint>{p});
  g.query(p);

  uint g2_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{three, p});
  uint p2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{g2_dist});
  g.observe(p2, 0.8);

  GlobalState state = GlobalState(g);

  Eigen::VectorXd unconstrained_values(1);
  unconstrained_values << -0.5;
  state.set_flattened_unconstrained_values(unconstrained_values);

  state.update_backgrad();
  Eigen::VectorXd grads1;
  state.get_flattened_unconstrained_grads(grads1);
  /*
  log_transform = dist.ExpTransform().inv
  g_dist = dist.TransformedDistribution(dist.Gamma(1, 2), log_transform)
  p = tensor(-0.5, requires_grad=True)
  g2_dist = dist.Gamma(3, p.exp())
  p2 = tensor(0.8, requires_grad=True)
  log_prob = g_dist.log_prob(p) + g2_dist.log_prob(p2)
  grad(log_prob, p)
  */
  EXPECT_NEAR(grads1[0], 2.3017, 0.001);
}

TEST(testglobal, global_state_gamma_transform) {
  /*
  Testing global state with a log-transformation of two variable,
  one depending on the other.

  We use the following model:
  p ~ Gamma(1, 2)   <- log transform
  p2 ~ Gamma(3, p)  <- log transform
  query p and p2
  */
  Graph g;
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint three = g.add_constant_pos_real(3.0);

  uint g_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{one, two});
  uint p = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{g_dist});
  g.query(p);

  uint g2_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{three, p});
  uint p2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{g2_dist});
  g.query(p2);
  g.customize_transformation(TransformType::LOG, {p, p2});

  GlobalState state = GlobalState(g);

  Eigen::VectorXd unconstrained_values(2);
  unconstrained_values << -0.5, -0.2231;
  state.set_flattened_unconstrained_values(unconstrained_values);

  state.update_backgrad();
  Eigen::VectorXd grads1;
  state.get_flattened_unconstrained_grads(grads1);
  /*
  log_transform = dist.ExpTransform().inv
  g_dist = dist.TransformedDistribution(dist.Gamma(1, 2), log_transform)
  p = tensor(-0.5, requires_grad=True)
  g2_dist = dist.TransformedDistribution(dist.Gamma(3, p.exp()), log_transform)
  p2 = tensor(-0.2231, requires_grad=True)

  log_prob = g_dist.log_prob(p) + g2_dist.log_prob(p2)
  grad(log_prob, p)
  */
  EXPECT_NEAR(grads1[0], 2.3017, 0.001);
}

TEST(testglobal, global_state_initialization) {
  /* Test initialization in real space */
  Graph g;
  uint hundred = g.add_constant(100.0);
  uint one = g.add_constant_pos_real(1.0);
  uint thousand = g.add_constant((natural_t)1000);

  uint normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{hundred, one});
  uint sample = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, thousand});
  g.query(sample);

  GlobalState state = GlobalState(g);
  uint seed = 17;

  // check all values = 0.0
  state.initialize_values(InitType::ZERO, seed);
  Eigen::VectorXd flattened_values;
  state.get_flattened_unconstrained_values(flattened_values);
  for (int i = 0; i < flattened_values.size(); i++) {
    EXPECT_NEAR(flattened_values[i], 0.0, 1e-4);
  }

  // check all values are between -2 and 2
  // values should be centered around 0 but nonzero
  state.initialize_values(InitType::RANDOM, seed);
  state.get_flattened_unconstrained_values(flattened_values);
  for (int i = 0; i < flattened_values.size(); i++) {
    EXPECT_GE(flattened_values[i], -2.0);
    EXPECT_LE(flattened_values[i], 2.0);
  }
  EXPECT_NEAR(flattened_values.mean(), 0.0, 0.1);
  EXPECT_FALSE(flattened_values.isZero());

  // check that values are drawn from the prior Normal(100, 1)
  state.initialize_values(InitType::PRIOR, seed);
  state.get_flattened_unconstrained_values(flattened_values);
  EXPECT_NEAR(flattened_values.mean(), 100.0, 0.1);
}

TEST(testglobal, global_state_transform_initialization) {
  /* Test initialization in log_transform space */
  Graph g;
  uint twohundred = g.add_constant_pos_real(200.0);
  uint hundred = g.add_constant_pos_real(100.0);
  uint thousand = g.add_constant((natural_t)1000);

  uint normal_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{twohundred, hundred});
  uint sample = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, thousand});
  g.query(sample);
  g.customize_transformation(TransformType::LOG, {sample});

  GlobalState state = GlobalState(g);
  uint seed = 17;

  // check all values = 0.0
  state.initialize_values(InitType::ZERO, seed);
  Eigen::VectorXd flattened_values;
  state.get_flattened_unconstrained_values(flattened_values);
  for (int i = 0; i < flattened_values.size(); i++) {
    EXPECT_NEAR(flattened_values[i], 0.0, 1e-4);
  }

  // check all values are between -2 and 2
  // values should be centered around 0 but nonzero
  state.initialize_values(InitType::RANDOM, seed);
  state.get_flattened_unconstrained_values(flattened_values);
  for (int i = 0; i < flattened_values.size(); i++) {
    EXPECT_GE(flattened_values[i], -2.0);
    EXPECT_LE(flattened_values[i], 2.0);
  }
  EXPECT_NEAR(flattened_values.mean(), 0.0, 0.1);
  EXPECT_FALSE(flattened_values.isZero());

  // check that values are drawn from the prior Gamma(200, 100)
  // mean of Gamma(200, 100): 2.0
  state.initialize_values(InitType::PRIOR, seed);
  state.get_flattened_unconstrained_values(flattened_values);
  EXPECT_NEAR(flattened_values.mean(), std::log(2.0), 0.1);
}
