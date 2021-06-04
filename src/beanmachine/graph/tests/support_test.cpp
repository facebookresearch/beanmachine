// Copyright (c) Facebook, Inc. and its affiliates.
#include <vector>

#include <gtest/gtest.h>

#include <beanmachine/graph/graph.h>

using namespace beanmachine;

TEST(testgraph, support) {
  graph::Graph g;
  // c1 and c2 are parents of o1 -> d1 -> o2
  uint c1 = g.add_constant_probability(.3);
  uint c2 = g.add_constant_probability(.4);
  uint o1 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c1, c2}));
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{o1});
  uint o2 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  // c3 and c4 are parents of o3 -> d2 -> o4
  uint c3 = g.add_constant_pos_real(.3);
  uint c4 = g.add_constant_pos_real(.4);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c3, c4}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{o3});
  uint o4 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  uint ro2 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o2}));
  uint ro4 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o4}));
  // o2 and o4 -> o5
  uint o5 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({ro2, ro4}));
  uint d3 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o5}));
  uint o6 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d3}));
  uint o7 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o6}));
  uint o8 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({o3, o7}));
  // observe o4 so support includes c3, c4, o3, d2, o4
  // however, we limit support to operators only: o3 and o4
  g.observe(o4, true);
  auto supp = g.compute_support();
  EXPECT_EQ(supp.size(), 2);
  EXPECT_EQ(*supp.begin(), o3);
  EXPECT_EQ(*supp.rbegin(), o4);
  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) =
      g.get_nodes_up_to_immediate_stochastic_descendants(o3, supp);
  // o3 -> det: o3, d2, o8 sto: o4
  // limiting to operators: o3 -> det: o3, o8 sto: o4
  // limiting to support: c3 -> det: o3 sto: o4
  EXPECT_EQ(det_nodes.size(), 1);
  EXPECT_EQ(det_nodes.front(), o3);
  EXPECT_EQ(sto_nodes.size(), 1);
  EXPECT_EQ(sto_nodes.front(), o4);
  std::vector<uint> det_anc;
  std::vector<uint> sto_anc;
  std::tie(det_anc, sto_anc) = g.compute_ancestors(o4);
  // o4 -ancestors-> det: c3, c4, o3, d2 sto:
  // restricting to operators: det: o3 sto:
  EXPECT_EQ(det_anc.size(), 1);
  EXPECT_EQ(det_anc.front(), o3);
  EXPECT_EQ(sto_anc.size(), 0);
  std::tie(det_anc, sto_anc) = g.compute_ancestors(o8);
  // restricting to operators, ancestors(o8) = det: o3, o7 sto: o6
  EXPECT_EQ(det_anc.size(), 2);
  EXPECT_EQ(det_anc.front(), o3);
  EXPECT_EQ(det_anc.back(), o7);
  EXPECT_EQ(sto_anc.size(), 1);
  EXPECT_EQ(sto_anc.front(), o6);
  // query o5 so support is now 13 nodes:
  //   c1, c2, o1, d1, o2, ro2, c3, c4, o3, d2, o4, ro5, o5
  // but we only include operators o1, o2, ro2, o3, o4, ro4, and o5
  g.query(o5);
  supp = g.compute_support();
  EXPECT_EQ(supp.size(), 7);
  // o4 -> det: o5, d3 sto: o4, o6
  // limiting to operators: o4 -> det: ro4, o5 sto: o4, o6
  // note: o7 and o8 are not in the descendants of o4
  // because the descendant chain gets cut off at the stochastic node o6
  // limiting to support: o4 -> det: ro4, o5 sto: o4
  std::tie(det_nodes, sto_nodes) =
      g.get_nodes_up_to_immediate_stochastic_descendants(o4, supp);
  EXPECT_EQ(det_nodes.size(), 2);
  EXPECT_EQ(det_nodes.front(), ro4);
  EXPECT_EQ(det_nodes.back(), o5);
  EXPECT_EQ(sto_nodes.size(), 1);
  EXPECT_EQ(sto_nodes.front(), o4);
  std::tie(det_anc, sto_anc) = g.compute_ancestors(o5);
  // ancestors(o5) = det: ro2, ro4, sto: o2 o4
  EXPECT_EQ(det_anc.size(), 2);
  EXPECT_EQ(det_anc.front(), ro2);
  EXPECT_EQ(det_anc.back(), ro4);
  EXPECT_EQ(sto_anc.size(), 2);
  EXPECT_EQ(sto_anc.front(), o2);
  EXPECT_EQ(sto_anc.back(), o4);
}
