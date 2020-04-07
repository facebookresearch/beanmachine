#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

#include "beanmachine/graph/proposer/gamma.h"

using namespace beanmachine::graph;
using namespace beanmachine::proposer;

TEST(testproposer, gamma) {
  const double ALPHA = 1;
  const double BETA = 0.1;
  Gamma dist(ALPHA, BETA);
  std::mt19937 gen(31425);
  uint num_samples = 10000;
  double sum = 0.0;
  double sumsq = 0.0;
  for (uint i=0; i<num_samples; i++) {
    auto val = dist.sample(gen);
    ASSERT_GT(val._double, 0.0);
    sum += val._double;
    sumsq += val._double * val._double;
  }
  double mean = sum / num_samples;
  double var = sumsq / num_samples - mean * mean;
  EXPECT_NEAR(mean, ALPHA/BETA, 0.1);
  EXPECT_NEAR(var, ALPHA / (BETA * BETA), 1.0);
}
