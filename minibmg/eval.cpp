/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/eval.h"
#include <beanmachine/minibmg/ad/real.h>
#include <beanmachine/minibmg/minibmg.h>
#include <random>

using namespace std;

namespace beanmachine::minibmg {

double sample_distribution(
    Operator distribution,
    function<double(uint)> get_parameter,
    mt19937& gen) {
  switch (distribution) {
    case Operator::DISTRIBUTION_NORMAL: {
      double mean = get_parameter(0);
      double stddev = get_parameter(1);
      normal_distribution<double> d{mean, stddev};
      return d(gen);
    }
    case Operator::DISTRIBUTION_BETA: {
      double a = get_parameter(0);
      double b = get_parameter(1);
      gamma_distribution<double> distrib_a(a, 1);
      gamma_distribution<double> distrib_b(b, 1);
      double x = distrib_a(gen);
      double y = distrib_b(gen);
      double sum = x + y;
      if (sum == 0.0) {
        throw EvalError(
            "sample_distribution has a degenerate " + to_string(distribution));
      }
      double p = x / sum;
      return p;
    }
    case Operator::DISTRIBUTION_BERNOULLI: {
      double probability_of_one = get_parameter(0);
      uniform_real_distribution<double> p(0.0, 1.0);
      bool result = p(gen) < probability_of_one;
      return double(result);
    }
    default:
      throw EvalError(
          "sample_distribution does not support " + to_string(distribution));
  }
}

} // namespace beanmachine::minibmg
