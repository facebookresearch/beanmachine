/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sstream>
#include <unordered_map>
#include "beanmachine/minibmg/ad/num3.h"
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/fluid_factory.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/localopt.h"
#include "beanmachine/minibmg/pretty.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

template <class T>
requires Number<T> T logit(const T& x) {
  return log(x / (1 - x));
}

TEST(localopt_test, symbolic_derivatives) {
  Graph::FluidFactory fac;
  auto f = beta(2, 2);
  auto s = sample(f);
  fac.query(s);
  auto c = bernoulli(s);
  fac.observe(sample(c), 1);
  fac.observe(sample(c), 1);
  fac.observe(sample(c), 0);
  Graph graph = fac.build();

  // Locate the node of interest.
  Nodep rv = graph.queries[0];

  // evaluate the graph's log prob, and its first and second derivatives.
  using T = Num3<Traced>;
  std::mt19937 gen;
  std::function<T(const std::string&, const unsigned int)> read_variable =
      [](const std::string& name, const unsigned int id) -> T {
    return Traced::variable(name, id);
  };
  std::unordered_map<Nodep, T> data;
  std::function<SampledValue<T>(
      const Distribution<T>& distribution, std::mt19937& gen)>
      sampler = [](const Distribution<T>& distribution,
                   std::mt19937&) -> SampledValue<T> {
    auto constrained = T{Traced::variable("rvid.constrained", 0), 1, 0};
    auto unconstrained = distribution.transformation()->call(constrained);
    auto log_prob = distribution.log_prob(constrained);
    return SampledValue<T>{constrained, unconstrained, log_prob};
  };
  auto eval_result = eval_graph(
      graph,
      gen,
      read_variable,
      data,
      /* run_queries= */ false,
      /* eval_log_prob= */ true,
      /* sampler= */ sampler);

  eval_result = opt(eval_result);
  auto log_prob = eval_result.log_prob.primal.node;
  auto d1 = eval_result.log_prob.derivative1.node;
  auto d2 = eval_result.log_prob.derivative2.node;
  std::vector<Nodep> roots = {log_prob, d1, d2};
  auto print_result = pretty_print(roots);

  std::stringstream printed;
  for (auto a : print_result.prelude) {
    printed << a << std::endl;
  }

  printed << "log_prob = ";
  printed << print_result.code[log_prob] << std::endl;
  printed << "d1 = ";
  printed << print_result.code[d1] << std::endl;
  printed << "d2 = ";
  printed << print_result.code[d2] << std::endl;

  auto expected = R"(auto temp_1 = -pow(rvid.constrained, -2);
auto temp_2 = 1 - rvid.constrained;
auto temp_3 = -pow(temp_2, -2);
auto temp_4 = 1 / rvid.constrained;
auto temp_5 = -1 / temp_2;
auto temp_6 = log(rvid.constrained);
auto temp_7 = log(temp_2);
log_prob = temp_6 + temp_7 + 1.791759469228055 + temp_6 + temp_6 + temp_7
d1 = temp_4 + temp_5 + temp_4 + temp_4 + temp_5
d2 = temp_1 + temp_3 + temp_1 + temp_1 + temp_3
)";
  ASSERT_EQ(expected, printed.str());
}
