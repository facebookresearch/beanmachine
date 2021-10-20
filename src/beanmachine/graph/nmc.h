// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/mh.h"

namespace beanmachine {
namespace graph {

class NMC : public MH {
 public:
  virtual ~NMC();

  NMC(Graph* graph, unsigned int seed);

  virtual std::string is_not_supported(Node* node) override;
};

} // namespace graph
} // namespace beanmachine
