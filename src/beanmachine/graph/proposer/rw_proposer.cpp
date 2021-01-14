// Copyright (c) Facebook, Inc. and its affiliates.
#include <math.h>
#include <random>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/beta.h"
#include "beanmachine/graph/proposer/delta.h"
#include "beanmachine/graph/proposer/gamma.h"
#include "beanmachine/graph/proposer/mixture.h"
#include "beanmachine/graph/proposer/normal.h"
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

const double MAIN_PROPOSER_WEIGHT = 1.0;
const double RANDOM_WALK_WEIGHT = 0.01;

std::unique_ptr<Proposer>
rw_proposer(const graph::NodeValue& value, double grad1, double grad2) {
  std::vector<double> weights;
  std::vector<std::unique_ptr<Proposer>> proposers;
  // For boolean variables, we will flip a fair coin
  if (value.type == graph::AtomicType::BOOLEAN) {
    weights.push_back(0.5);
    proposers.push_back(
        std::make_unique<Delta>(graph::NodeValue(not value._bool)));
    weights.push_back(0.5);
    proposers.push_back(std::make_unique<Delta>(graph::NodeValue(value._bool)));
  }
  // For continuous-valued variables we will use a random walk
  if (value.type == graph::AtomicType::PROBABILITY) {
    double x = value._double;
    assert(x > 0 and x < 1);
    // a random walk for a probability is a Beta proposer with strength one
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<Beta>(x, 1 - x));
  } else if (value.type == graph::AtomicType::REAL) {
    double x = value._double;
    // a random walk from a standard normal
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<Normal>(x, 1.0));
  } else if (value.type == graph::AtomicType::POS_REAL) {
    double x = value._double;
    // A random walk using an Exponential distribution centered at the
    // current value
    weights.push_back(RANDOM_WALK_WEIGHT);
    proposers.push_back(std::make_unique<Gamma>(1.0, 1.0 / x));
  }
  return std::make_unique<Mixture>(weights, std::move(proposers));
}

} // namespace proposer
} // namespace beanmachine
