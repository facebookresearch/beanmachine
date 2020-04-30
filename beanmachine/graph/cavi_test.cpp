// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>
#include <cmath>
#include <tuple>

#include <gtest/gtest.h>

#include <beanmachine/graph/graph.h>

using namespace beanmachine;

TEST(testcavi, noisy_or) {
  // see cavi_test.py:build_graph2 for an explanation of this model
  graph::Graph g;
  uint c_prior = g.add_constant_probability(0.01);
  uint d_prior = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c_prior}));
  uint x =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d_prior}));
  uint y =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d_prior}));
  uint pos_x =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({x}));
  uint pos_y =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({y}));
  uint c_m_log_pt01 = g.add_constant_pos_real(-log(0.01));
  uint c_m_log_pt99 = g.add_constant_pos_real(-log(0.99));
  uint param = g.add_operator(
      graph::OperatorType::ADD,
      std::vector<uint>({
          c_m_log_pt99,
          g.add_operator(
              graph::OperatorType::MULTIPLY,
              std::vector<uint>({c_m_log_pt01, pos_x})),
          g.add_operator(
              graph::OperatorType::MULTIPLY,
              std::vector<uint>({c_m_log_pt01, pos_y})),
      }));
  uint d_like = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({param}));
  uint z =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d_like}));
  g.observe(z, true);
  g.query(x);
  g.query(y);
  // run CAVI on the above graph and verify the results
  const std::vector<std::vector<double>>& parameters =
      g.variational(100, 1000, 81391, 1000);
  EXPECT_NEAR(parameters[0][0], parameters[1][0], 0.1);
  EXPECT_NEAR(parameters[0][0], 0.245, 0.1);
  const auto& elbo = g.get_elbo();
  EXPECT_NEAR(elbo.back(), -3.7867, 0.1);
}
