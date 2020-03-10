// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"


using namespace beanmachine::graph;

TEST(testrejection, beta_bernoulli) {
  Graph g;
  uint a = g.add_constant(2.0);
  uint b = g.add_constant(3.0);
  uint prior = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({prior}));
  uint n = g.add_constant((natural_t) 5);
  uint like = g.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>({n, prob})
  );
  uint k = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({like}));
  g.observe(k, (natural_t) 2);
  g.query(prob);
  auto& means = g.infer_mean(1000, InferenceType::REJECTION, 23891);
  EXPECT_NEAR(means[0], 0.4, 1e-2);
}
