// Copyright (c) Facebook, Inc. and its affiliates.
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

void Graph::rw(uint num_samples, std::mt19937& gen) {
  Graph::_mh(num_samples, gen, proposer::rw_proposer);
}
} // namespace graph
} // namespace beanmachine
