/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <beanmachine/minibmg/graph_factory.h>
#include <beanmachine/minibmg/tests/test_utils.h>
#include <gtest/gtest.h>
#include <random>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/num3.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/eval.h"

// using namespace ::testing;
using namespace ::beanmachine::minibmg;
using namespace ::std;

TEST(eval_test, simple1) {
  // a simple graph that evaluates a few operators is evaluated and the final
  // result is compared to its expected value.
  Graph::Factory fac;
  auto k0 = fac.constant(1.2); // 1.2
  auto k1 = fac.constant(4.1); // 4.1
  auto add0 = fac.add(k0, k1); // 5.3
  auto v1 = fac.variable("x", 0); // 1.15
  auto mul1 = fac.multiply(add0, v1); // 6.095
  auto sub1 = fac.subtract(mul1, k1); // 1.995
  fac.query(sub1); // add a root to the graph.
  auto graph = fac.build();
  std::mt19937 gen{123456};
  auto read_variable = [](const std::string&, const unsigned) { return 1.15; };
  int graph_size = graph.size();
  unordered_map<Nodep, Real> data;
  auto eval_result = eval_graph<Real>(
      graph, gen, read_variable, data, /* run_queries= */ true);
  EXPECT_CLOSE(1.995, eval_result.queries[0]);
}

TEST(eval_test, sample1) {
  // a graph that produces normal samples is evaluated many times
  // and the statistics of the samples are compared to their expected values.
  std::exception x1;
  Graph::Factory fac;
  double expected_mean = 12.34;
  double expected_stdev = 5.67;
  auto k0 = fac.constant(expected_mean);
  auto k1 = fac.constant(expected_stdev);
  auto normal0 = fac.normal(k0, k1);
  auto sample0 = fac.sample(normal0);
  fac.query(sample0); // add a root to the graph.
  auto graph = fac.build();
  // We create a new random number generator with a specific seed so that this
  // test will be deterministic and not be flaky.
  std::mt19937 gen{123456};
  // Evaluate the graph many times and gather stats.
  int n = 10000;
  double sum = 0;
  double sum_squared = 0;
  int graph_size = graph.size();
  std::unordered_map<Nodep, Real> data;
  for (int i = 0; i < n; i++) {
    auto eval_result =
        eval_graph<Real>(graph, gen, nullptr, data, /* run_queries= */ true);
    auto sample = eval_result.queries[0];
    sum += sample;
    sum_squared += sample * sample;
  }

  double mean = sum / n;
  double variance = sum_squared / n - (mean * mean);
  double stdev = std::sqrt(variance);
  EXPECT_CLOSE_EPS(expected_mean, mean, 0.01);
  EXPECT_CLOSE_EPS(expected_stdev, stdev, 0.01);
}

// the function f
template <class T>
requires Number<T> T f(T x) {
  return 1.1 * pow(x, 2);
}

// f's first derivative
template <class T>
requires Number<T> T fp(T x) {
  return 2.2 * x;
}

// f's second derivative
template <class T>
requires Number<T> T fpp(T) {
  return 2.2;
}

using Dual = Num2<Real>;

TEST(eval_test, derivative_dual) {
  // a graph that computes a function of a variable, so we can compute
  // the derivative with respect to that variable.
  std::mt19937 gen{123456};
  Graph::Factory fac;
  auto s = fac.multiply(
      fac.constant(1.1), fac.pow(fac.variable("x", 0), fac.constant(2)));
  fac.query(s); // add a root to the graph.
  Graph graph = fac.build();
  int graph_size = graph.size();

  // We generate several doubles between -2.0 and 2.0 to test with.
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  std::unordered_map<Nodep, Dual> data;
  for (int i = 0; i < 10; i++) {
    double input = unif(gen);
    auto read_variable = [=](const std::string&, const unsigned) {
      return Dual{input, 1};
    };
    data.clear();
    auto eval_result = eval_graph<Dual>(
        graph, gen, read_variable, data, /* run_queries = */ true);
    Nodep s_node = fac[s];
    EXPECT_CLOSE(f<Real>(input).as_double(), data[s_node].primal.as_double());
    EXPECT_CLOSE(
        fp<Real>(input).as_double(), data[s_node].derivative1.as_double());
  }
}

using Triune = Num3<Real>;

TEST(eval_test, derivatives_triune) {
  // a graph that computes a function of a variable, so we can compute
  // the first and second derivatives with respect to that variable.
  std::mt19937 gen{123456};
  Graph::Factory fac;
  auto s = fac.multiply(
      fac.constant(1.1), fac.pow(fac.variable("x", 0), fac.constant(2)));
  fac.query(s); // add a root to the graph.
  auto sn = fac[s];
  Graph graph = fac.build();
  int graph_size = graph.size();

  // We generate several doubles between -2.0 and 2.0 to test with.
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  std::unordered_map<Nodep, Triune> data;
  for (int i = 0; i < 10; i++) {
    double input = unif(gen);
    auto read_variable = [=](const std::string&, const unsigned) {
      return Triune{input, 1, 0};
    };
    data.clear();
    eval_graph<Triune>(graph, gen, read_variable, data);
    EXPECT_CLOSE(f<Real>(input).as_double(), data[sn].primal.as_double());
    EXPECT_CLOSE(fp<Real>(input).as_double(), data[sn].derivative1.as_double());
    EXPECT_CLOSE(
        fpp<Real>(input).as_double(), data[sn].derivative2.as_double());
  }
}

TEST(eval_test, log1p) {
  Graph::Factory fac;
  auto k = 1.1;
  auto kn = fac.constant(1.1);
  auto log1p = fac.log1p(kn);
  auto qi = fac.query(log1p);
  auto graph = fac.build();
  std::mt19937 gen{123456};
  auto read_variable = [=](const std::string&, const unsigned) -> Real {
    throw nullptr; // this model has no variables to read
  };
  std::unordered_map<Nodep, Real> data;
  auto eval_result = eval_graph<Real>(
      graph, gen, read_variable, data, /* run_queries = */ true);
  EXPECT_CLOSE(eval_result.queries[qi], std::log1p(k));
}
