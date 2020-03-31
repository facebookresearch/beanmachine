// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

using namespace beanmachine;

TEST(testoperator, complement) {
  // negative test num args can't be zero
  EXPECT_THROW(
    oper::Operator onode1(
        graph::OperatorType::COMPLEMENT, std::vector<graph::Node*>{}),
    std::invalid_argument);
  auto p1 = graph::AtomicValue(graph::AtomicType::PROBABILITY, 0.1);
  graph::ConstNode cnode1(p1);
  // negative test num args can't be two
  EXPECT_THROW(
    oper::Operator(
        graph::OperatorType::COMPLEMENT, std::vector<graph::Node*>{&cnode1, &cnode1}),
    std::invalid_argument);
  auto r1 = graph::AtomicValue(graph::AtomicType::REAL, 0.1);
  graph::ConstNode cnode2(r1);
  // negative test arg can't be real
  EXPECT_THROW(
    oper::Operator(
        graph::OperatorType::COMPLEMENT, std::vector<graph::Node*>{&cnode2}),
    std::invalid_argument);
  // complement of prob is 1-prob
  oper::Operator onode1(
      graph::OperatorType::COMPLEMENT, std::vector<graph::Node*>{&cnode1});
  EXPECT_EQ(onode1.value.type, graph::AtomicType::PROBABILITY);
  onode1.in_nodes.push_back(&cnode1);
  std::mt19937 generator(31245);
  onode1.eval(generator);
  EXPECT_NEAR(onode1.value._double, 0.9, 0.001);
  // complement of bool is logical_not(bool)
  auto b1 = graph::AtomicValue(false);
  graph::ConstNode cnode3(b1);
  oper::Operator onode2(
      graph::OperatorType::COMPLEMENT, std::vector<graph::Node*>{&cnode3});
  EXPECT_EQ(onode2.value.type, graph::AtomicType::BOOLEAN);
  onode2.in_nodes.push_back(&cnode3);
  onode2.eval(generator);
  EXPECT_EQ(onode2.value._bool, true);
}
