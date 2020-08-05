// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/bernoulli_noisy_or.h"
#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/distribution/binomial.h"
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/distribution/tabular.h"
#include "beanmachine/graph/distribution/flat.h"
#include "beanmachine/graph/distribution/normal.h"
#include "beanmachine/graph/distribution/half_cauchy.h"
#include "beanmachine/graph/distribution/student_t.h"
#include "beanmachine/graph/distribution/bernoulli_logit.h"
#include "beanmachine/graph/distribution/gamma.h"

namespace beanmachine {
namespace distribution {

std::unique_ptr<Distribution> Distribution::new_distribution(
    graph::DistributionType dist_type,
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes) {
  // check parent nodes are of the correct type
  for (graph::Node* parent : in_nodes) {
    if (parent->node_type == graph::NodeType::DISTRIBUTION) {
      throw std::invalid_argument("distribution parents can't be distribution");
    }
  }
  // now simply call the appropriate distribution constructor
  switch(dist_type) {
    case graph::DistributionType::TABULAR: {
      return std::make_unique<Tabular>(sample_type, in_nodes);
    }
    case graph::DistributionType::BERNOULLI: {
      return std::make_unique<Bernoulli>(sample_type, in_nodes);
    }
    case graph::DistributionType::BERNOULLI_NOISY_OR: {
      return std::make_unique<BernoulliNoisyOr>(sample_type, in_nodes);
    }
    case graph::DistributionType::BETA: {
      return std::make_unique<Beta>(sample_type, in_nodes);
    }
    case graph::DistributionType::BINOMIAL: {
      return std::make_unique<Binomial>(sample_type, in_nodes);
    }
    case graph::DistributionType::FLAT: {
      return std::make_unique<Flat>(sample_type, in_nodes);
    }
    case graph::DistributionType::NORMAL: {
      return std::make_unique<Normal>(sample_type, in_nodes);
    }
    case graph::DistributionType::HALF_CAUCHY: {
      return std::make_unique<HalfCauchy>(sample_type, in_nodes);
    }
    case graph::DistributionType::STUDENT_T: {
      return std::make_unique<StudentT>(sample_type, in_nodes);
    }
    case graph::DistributionType::BERNOULLI_LOGIT: {
      return std::make_unique<BernoulliLogit>(sample_type, in_nodes);
    }
    case graph::DistributionType::GAMMA: {
      return std::make_unique<Gamma>(sample_type, in_nodes);
    }
    default: {
      throw std::invalid_argument(
        "Unknown distribution " + std::to_string(static_cast<int>(dist_type)));
    }
  }
}

} // namespace distribution
} // namespace beanmachine
