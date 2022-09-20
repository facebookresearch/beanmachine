/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <beanmachine/minibmg/ad/tests/test_utils.h>
#include <beanmachine/minibmg/minibmg.h>
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
  auto k0 = fac.add_constant(1.2); // 1.2
  auto k1 = fac.add_constant(4.1); // 4.1
  auto add0 = fac.add_operator(Operator::ADD, {k0, k1}); // 5.3
  auto v1 = fac.add_variable("x", 0); // 1.15
  auto mul1 = fac.add_operator(Operator::MULTIPLY, {add0, v1}); // 6.095
  auto sub1 = fac.add_operator(Operator::SUBTRACT, {mul1, k1}); // 1.995
  auto graph = fac.build();
  std::mt19937 gen;
  auto read_variable = [](const std::string&, const unsigned) { return 1.15; };
  int graph_size = graph.size();
  vector<Real> data;
  data.assign(graph_size, 0);
  eval_graph<Real>(graph, gen, read_variable, data);
  EXPECT_CLOSE(1.995, data[sub1].as_double());
}

TEST(eval_test, sample1) {
  // a graph that produces normal samples is evaluated many times
  // and the statistics of the samples are compared to their expected values.
  Graph::Factory fac;
  double expected_mean = 12.34;
  double expected_stdev = 41.78;
  auto k0 = fac.add_constant(expected_mean);
  auto k1 = fac.add_constant(expected_stdev);
  auto normal0 = fac.add_operator(Operator::DISTRIBUTION_NORMAL, {k0, k1});
  auto sample0 = fac.add_operator(Operator::SAMPLE, {normal0});
  auto graph = fac.build();
  // We create a new random number generator with its default (deterministic)
  // seed so that this test will not be flaky.
  std::mt19937 gen;
  // Evaluate the graph many times and gather stats.
  int n = 10000;
  double sum = 0;
  double sum_squared = 0;
  int graph_size = graph.size();
  vector<Real> data;
  data.assign(graph_size, 0);
  for (int i = 0; i < n; i++) {
    eval_graph<Real>(graph, gen, nullptr, data);
    auto sample = data[sample0].as_double();
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
  std::mt19937 gen;
  Graph::Factory fac;
  auto s = fac.add_operator(
      Operator::MULTIPLY,
      {fac.add_constant(1.1),
       fac.add_operator(
           Operator::POW, {fac.add_variable("x", 0), fac.add_constant(2)})});
  Graph graph = fac.build();
  int graph_size = graph.size();

  // We generate several doubles between -2.0 and 2.0 to test with.
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  vector<Dual> data;
  for (int i = 0; i < 10; i++) {
    double input = unif(gen);
    auto read_variable = [=](const std::string&, const unsigned) {
      return Dual{input, 1};
    };
    data.clear();
    data.assign(graph_size, 0);
    eval_graph<Dual>(graph, gen, read_variable, data);
    EXPECT_CLOSE(f<Real>(input).as_double(), data[s].primal.as_double());
    EXPECT_CLOSE(fp<Real>(input).as_double(), data[s].derivative1.as_double());
  }
}

using Triune = Num3<Real>;

TEST(eval_test, derivatives_triune) {
  // a graph that computes a function of a variable, so we can compute
  // the first and second derivatives with respect to that variable.
  std::mt19937 gen;
  Graph::Factory fac;
  auto s = fac.add_operator(
      Operator::MULTIPLY,
      {fac.add_constant(1.1),
       fac.add_operator(
           Operator::POW, {fac.add_variable("x", 0), fac.add_constant(2)})});
  Graph graph = fac.build();
  int graph_size = graph.size();

  // We generate several doubles between -2.0 and 2.0 to test with.
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  vector<Triune> data;
  for (int i = 0; i < 10; i++) {
    double input = unif(gen);
    auto read_variable = [=](const std::string&, const unsigned) {
      return Triune{input, 1, 0};
    };
    data.clear();
    data.assign(graph_size, 0);
    eval_graph<Triune>(graph, gen, read_variable, data);
    EXPECT_CLOSE(f<Real>(input).as_double(), data[s].primal.as_double());
    EXPECT_CLOSE(fp<Real>(input).as_double(), data[s].derivative1.as_double());
    EXPECT_CLOSE(fpp<Real>(input).as_double(), data[s].derivative2.as_double());
  }
}
