#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

#include "beanmachine/graph/proposer/trunc_cauchy.h"

using namespace beanmachine::graph;
using namespace beanmachine::proposer;

TEST(testproposer, trunc_cauchy) {
  // all samples should be positive and the fraction between 0 and location can be exactly computed
  const double LOC = 5;
  const double SCALE = 10;
  TruncatedCauchy dist(LOC, SCALE);
  std::mt19937 gen(31425);
  uint num_samples = 10000;
  uint below_loc = 0;
  for (uint i=0; i<num_samples; i++) {
    auto val = dist.sample(gen);
    ASSERT_GT(val._double, 0.0);
    if (val._double < LOC) {
      below_loc ++;
    }
  }
  double atan_infty = M_PI_2;
  double atan_loc = std::atan((LOC-LOC)/SCALE);
  double atan_0 = std::atan((0-LOC)/SCALE);
  EXPECT_NEAR(double(below_loc)/num_samples, (atan_loc - atan_0)/(atan_infty - atan_0), 0.01);
  // log_prob at LOC should be higher than log_prob at zero
  AtomicValue val_0(AtomicType::POS_REAL, 0);
  AtomicValue val_loc(AtomicType::POS_REAL, LOC);
  EXPECT_GT(dist.log_prob(val_loc), dist.log_prob(val_0));
}
