// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

using namespace beanmachine;
using namespace beanmachine::graph;

TEST(testoperator, complement) {
  // negative test num args can't be zero
  EXPECT_THROW(
    oper::Operator onode1(
        OperatorType::COMPLEMENT, std::vector<Node*>{}),
    std::invalid_argument);
  auto p1 = AtomicValue(AtomicType::PROBABILITY, 0.1);
  ConstNode cnode1(p1);
  // negative test num args can't be two
  EXPECT_THROW(
    oper::Operator(
        OperatorType::COMPLEMENT, std::vector<Node*>{&cnode1, &cnode1}),
    std::invalid_argument);
  auto r1 = AtomicValue(AtomicType::REAL, 0.1);
  ConstNode cnode2(r1);
  // negative test arg can't be real
  EXPECT_THROW(
    oper::Operator(
        OperatorType::COMPLEMENT, std::vector<Node*>{&cnode2}),
    std::invalid_argument);
  // complement of prob is 1-prob
  oper::Operator onode1(
      OperatorType::COMPLEMENT, std::vector<Node*>{&cnode1});
  EXPECT_EQ(onode1.value.type, AtomicType::PROBABILITY);
  onode1.in_nodes.push_back(&cnode1);
  std::mt19937 generator(31245);
  onode1.eval(generator);
  EXPECT_NEAR(onode1.value._double, 0.9, 0.001);
  // complement of bool is logical_not(bool)
  auto b1 = AtomicValue(false);
  ConstNode cnode3(b1);
  oper::Operator onode2(
      OperatorType::COMPLEMENT, std::vector<Node*>{&cnode3});
  EXPECT_EQ(onode2.value.type, AtomicType::BOOLEAN);
  onode2.in_nodes.push_back(&cnode3);
  onode2.eval(generator);
  EXPECT_EQ(onode2.value._bool, true);
}

TEST(testoperator, phi) {
  Graph g;
  // negative tests: exactly one real should be the input to a PHI
  EXPECT_THROW(
    g.add_operator(OperatorType::PHI, std::vector<uint>{}),
    std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
    g.add_operator(OperatorType::PHI, std::vector<uint>{pos1}),
    std::invalid_argument);
  auto real1 = g.add_constant(0.0);
  EXPECT_THROW(
    g.add_operator(OperatorType::PHI, std::vector<uint>{real1, real1}),
    std::invalid_argument);
  // y ~ Bernoulli(Phi(x^2))  and x = 0.5; note: Phi(0.25) = 0.5987
  auto prior = g.add_distribution(
    DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto phi_x_sq = g.add_operator(OperatorType::PHI, std::vector<uint>{x_sq});
  auto likelihood = g.add_distribution(
    DistributionType::BERNOULLI, AtomicType::BOOLEAN, std::vector<uint>{phi_x_sq});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::GIBBS);
  EXPECT_NEAR(means[0], 0.5987, 0.01);
  g.observe(y, true);
  // check gradient:
  // f(x) = log(Phi(x)); want to check f'(x) and f''(x)
  // Verified in pytorch using the following code:
  // x = torch.tensor([0.5], requires_grad=True)
  // f_x = torch.log(torch.distributions.normal.Normal(0,1).cdf(x**2))
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> 0.6458 and f_grad2 -> 0.7131
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6458, 1e-3);
  EXPECT_NEAR(grad2, 0.7131, 1e-3);
}
