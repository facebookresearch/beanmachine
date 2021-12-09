/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/bernoulli_logit.h"
#include "beanmachine/graph/distribution/bernoulli_noisy_or.h"
#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/distribution/bimixture.h"
#include "beanmachine/graph/distribution/binomial.h"
#include "beanmachine/graph/distribution/categorical.h"
#include "beanmachine/graph/distribution/dirichlet.h"
#include "beanmachine/graph/distribution/flat.h"
#include "beanmachine/graph/distribution/gamma.h"
#include "beanmachine/graph/distribution/half_cauchy.h"
#include "beanmachine/graph/distribution/half_normal.h"
#include "beanmachine/graph/distribution/normal.h"
#include "beanmachine/graph/distribution/student_t.h"
#include "beanmachine/graph/distribution/tabular.h"

namespace beanmachine {
namespace distribution {

std::unique_ptr<Distribution> Distribution::new_distribution(
    graph::DistributionType dist_type,
    graph::ValueType sample_type,
    const std::vector<graph::Node*>& in_nodes) {
  // call the appropriate distribution constructor
  if (sample_type.variable_type == graph::VariableType::SCALAR) {
    auto atype = sample_type.atomic_type;
    switch (dist_type) {
      case graph::DistributionType::TABULAR: {
        return std::make_unique<Tabular>(atype, in_nodes);
      }
      case graph::DistributionType::BERNOULLI: {
        return std::make_unique<Bernoulli>(atype, in_nodes);
      }
      case graph::DistributionType::BERNOULLI_NOISY_OR: {
        return std::make_unique<BernoulliNoisyOr>(atype, in_nodes);
      }
      case graph::DistributionType::BETA: {
        return std::make_unique<Beta>(atype, in_nodes);
      }
      case graph::DistributionType::BINOMIAL: {
        return std::make_unique<Binomial>(atype, in_nodes);
      }
      case graph::DistributionType::FLAT: {
        return std::make_unique<Flat>(atype, in_nodes);
      }
      case graph::DistributionType::NORMAL: {
        return std::make_unique<Normal>(atype, in_nodes);
      }
      case graph::DistributionType::HALF_NORMAL: {
        return std::make_unique<Half_Normal>(atype, in_nodes);
      }
      case graph::DistributionType::HALF_CAUCHY: {
        return std::make_unique<HalfCauchy>(atype, in_nodes);
      }
      case graph::DistributionType::STUDENT_T: {
        return std::make_unique<StudentT>(atype, in_nodes);
      }
      case graph::DistributionType::BERNOULLI_LOGIT: {
        return std::make_unique<BernoulliLogit>(atype, in_nodes);
      }
      case graph::DistributionType::GAMMA: {
        return std::make_unique<Gamma>(atype, in_nodes);
      }
      case graph::DistributionType::BIMIXTURE: {
        return std::make_unique<Bimixture>(atype, in_nodes);
      }
      case graph::DistributionType::CATEGORICAL: {
        return std::make_unique<Categorical>(atype, in_nodes);
      }
      default: {
        throw std::invalid_argument(
            "Unknown distribution " +
            std::to_string(static_cast<int>(dist_type)) +
            " for univariate sample type.");
      }
    }
  } else if (
      sample_type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX) {
    switch (dist_type) {
      case graph::DistributionType::DIRICHLET: {
        return std::make_unique<Dirichlet>(sample_type, in_nodes);
      }
      default: {
        throw std::invalid_argument(
            "Unknown distribution " +
            std::to_string(static_cast<int>(dist_type)) +
            " for multivariate sample type.");
      }
    }
  } else {
    switch (dist_type) {
      case graph::DistributionType::FLAT: {
        return std::make_unique<Flat>(sample_type, in_nodes);
      }
      default: {
        throw std::invalid_argument(
            "Unknown distribution " +
            std::to_string(static_cast<int>(dist_type)) +
            " for multivariate sample type.");
      }
    }
  }
}

graph::NodeValue Distribution::sample(std::mt19937& gen) const {
  auto sample_value = graph::NodeValue(sample_type);
  this->sample(gen, sample_value);
  return sample_value;
}

void Distribution::sample(std::mt19937& gen, graph::NodeValue& sample_value)
    const {
  // sample a single SCALAR
  if (sample_value.type.variable_type == graph::VariableType::SCALAR) {
    switch (sample_value.type.atomic_type) {
      case graph::AtomicType::BOOLEAN:
        sample_value._bool = _bool_sampler(gen);
        break;
      case graph::AtomicType::REAL:
      case graph::AtomicType::POS_REAL:
      case graph::AtomicType::PROBABILITY:
        sample_value._double = _double_sampler(gen);
        break;
      case graph::AtomicType::NATURAL:
        sample_value._natural = _natural_sampler(gen);
        break;
      default:
        throw std::runtime_error("Unsupported sample type.");
        break;
    }
    return;
  }
  // iid sample SCALARs
  if (sample_type.variable_type == graph::VariableType::SCALAR and
      sample_value.type.variable_type ==
          graph::VariableType::BROADCAST_MATRIX) {
    uint size = sample_value.type.cols * sample_value.type.rows;
    assert(size > 1);
    switch (sample_value.type.atomic_type) {
      case graph::AtomicType::BOOLEAN:
        for (uint i = 0; i < size; i++) {
          *(sample_value._bmatrix.data() + i) = _bool_sampler(gen);
        }
        break;
      case graph::AtomicType::REAL:
      case graph::AtomicType::POS_REAL:
      case graph::AtomicType::PROBABILITY:
        for (uint i = 0; i < size; i++) {
          *(sample_value._matrix.data() + i) = _double_sampler(gen);
        }
        break;
      case graph::AtomicType::NATURAL:
        for (uint i = 0; i < size; i++) {
          *(sample_value._nmatrix.data() + i) = _natural_sampler(gen);
        }
        break;
      default:
        throw std::runtime_error("Unsupported sample type.");
        break;
    }
    return;
  } else if (
      sample_type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX) {
    switch (sample_type.atomic_type) {
      case graph::AtomicType::PROBABILITY:
        sample_value._matrix = _matrix_sampler(gen);
        break;
      default:
        throw std::runtime_error("Unsupported sample type.");
        break;
    }
    return;
  } else if (
      sample_type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    switch (sample_type.atomic_type) {
      case graph::AtomicType::REAL:
      case graph::AtomicType::POS_REAL:
      case graph::AtomicType::PROBABILITY:
        sample_value._matrix = _matrix_sampler(gen);
        break;
      default:
        throw std::runtime_error("Unsupported sample type.");
        break;
    }
    return;
  }
  throw std::runtime_error("Unsupported sample type.");
}

} // namespace distribution
} // namespace beanmachine
