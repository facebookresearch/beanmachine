/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <gtest/gtest.h>
#include <math.h>
#include <mcheck.h>
#include <vector>

#include "beanmachine/graph/graph.h"

using namespace ::testing;

namespace beanmachine {
namespace graph {

TEST(testgraph_stats, simple_graph_stats) {
  graph::Graph g;
  uint c1 = g.add_constant_probability(0.1);
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint o1 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  uint o2 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o1}));
  uint c2 = g.add_constant_pos_real(0.8);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c2, o2}));
  uint c3 = g.add_constant_pos_real(0.1);
  uint o4 =
      g.add_operator(graph::OperatorType::ADD, std::vector<uint>({c3, o3}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o4}));
  g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));

  std::string expected_output("Graph Statistics Report\n");
  expected_output += "#######################\n";
  expected_output += "Number of nodes: 10\n";
  expected_output += "Number of edges: 9\n";
  expected_output += "Graph density: 0.10\n";
  expected_output += "Number of root nodes: 3\n";
  expected_output += "Number of terminal nodes: 1\n";
  expected_output += "Maximum no. of incoming edges into a node: 2\n";
  expected_output += "Maximum no. of outgoing edges from a node: 1\n";
  expected_output += "\n";
  expected_output += "Node statistics:\n";
  expected_output += "################\n";
  expected_output += "CONSTANT: 3\n";
  expected_output += "\tRoot nodes: 3\n";
  expected_output += "\tConstant node statistics:\n";
  expected_output += "\t-------------------------\n";
  expected_output += "\t\tPROBABILITY and SCALAR: 1\n";
  expected_output += "\t\tPOS_REAL and SCALAR: 2\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 3\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 1 edges: 3\n";
  expected_output += "\n";
  expected_output += "DISTRIBUTION: 2\n";
  expected_output += "\tNo root or terminal nodes\n";
  expected_output += "\tDistribution node statistics:\n";
  expected_output += "\t-----------------------------\n";
  expected_output += "\t\tBERNOULLI: 1\n";
  expected_output += "\t\tBERNOULLI_NOISY_OR: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 1 edges: 2\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 1 edges: 2\n";
  expected_output += "\n";
  expected_output += "OPERATOR: 5\n";
  expected_output += "\tTerminal nodes: 1\n";
  expected_output += "\tOperator node statistics:\n";
  expected_output += "\t-------------------------\n";
  expected_output += "\t\tSAMPLE: 2\n";
  expected_output += "\t\tTO_POS_REAL: 1\n";
  expected_output += "\t\tMULTIPLY: 1\n";
  expected_output += "\t\tADD: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 1 edges: 3\n";
  expected_output += "\t\tNodes with 2 edges: 2\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 1\n";
  expected_output += "\t\tNodes with 1 edges: 4\n";
  expected_output += "\n";
  expected_output += "Edge statistics:\n";
  expected_output += "################\n";
  expected_output += "\tDistribution of incoming edges:\n";
  expected_output += "\t-------------------------------\n";
  expected_output += "\tNodes with 0 edges: 3\n";
  expected_output += "\tNodes with 1 edges: 5\n";
  expected_output += "\tNodes with 2 edges: 2\n";
  expected_output += "\n";
  expected_output += "\tDistribution of outgoing edges:\n";
  expected_output += "\t-------------------------------\n";
  expected_output += "\tNodes with 0 edges: 1\n";
  expected_output += "\tNodes with 1 edges: 9\n";
  expected_output += "\n";

  ASSERT_EQ(expected_output, g.collect_statistics());
}

TEST(testgraph_stats, all_stats) {
  // This is not a valid model, but a fictitious construct to test
  // the stats package against all possible node types
  graph::Graph g;
  // constants
  g.add_constant(true);
  uint c_natural_1 = g.add_constant((graph::natural_t)1);
  uint c_natural_2 = g.add_constant((graph::natural_t)2);
  g.add_constant((graph::natural_t)42);
  uint c_prob = g.add_constant_probability(0.5);
  uint c_real0 = g.add_constant(-2.5);
  uint c_real1 = g.add_constant(-42.01);
  g.add_constant(42.42);
  uint c_neg = g.add_constant_neg_real(-1.5);
  uint c_pos = g.add_constant_pos_real(2.5);
  uint c_pos1 = g.add_constant_pos_real(1.42);

  Eigen::MatrixXd m0 = Eigen::MatrixXd::Constant(2, 1, 0.6);
  g.add_constant_probability_matrix(m0);
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Identity(2, 2);
  g.add_constant_pos_matrix(m1);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(2, 2);
  g.add_constant_real_matrix(m2);
  Eigen::MatrixXd m3(2, 1);
  m3 << 0.2, 0.8;
  uint c3 = g.add_constant_col_simplex_matrix(m3);
  Eigen::MatrixXb m4(1, 2);
  m4 << true, false;
  g.add_constant_bool_matrix(m4);
  Eigen::MatrixXn m5(2, 1);
  m5 << 1, 2;
  g.add_constant_natural_matrix(m5);
  Eigen::MatrixXd m6(2, 1);
  m6 << 1.0, 2.0;
  g.add_constant_pos_matrix(m1);

  // distributions

  uint d_bernoulli = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_prob});
  g.add_distribution(
      graph::DistributionType::BERNOULLI_LOGIT,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_real0});
  g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_pos});
  uint d_beta = g.add_distribution(
      graph::DistributionType::BETA,
      graph::AtomicType::PROBABILITY,
      std::vector<uint>{c_pos, c_pos});
  g.add_distribution(
      graph::DistributionType::BIMIXTURE,
      graph::AtomicType::PROBABILITY,
      std::vector<uint>{c_prob, d_beta, d_beta});
  uint d_binomial = g.add_distribution(
      graph::DistributionType::BINOMIAL,
      graph::AtomicType::NATURAL,
      std::vector<uint>{c_natural_2, c_prob});
  g.add_distribution(
      graph::DistributionType::CATEGORICAL,
      graph::AtomicType::NATURAL,
      std::vector<uint>{c3});
  g.add_distribution(
      graph::DistributionType::CAUCHY,
      graph::AtomicType::REAL,
      std::vector<uint>{c_real0, c_pos});
  g.add_distribution(
      graph::DistributionType::FLAT,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{});
  uint d_gamma = g.add_distribution(
      graph::DistributionType::GAMMA,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{c_pos, c_pos1});
  g.add_distribution(
      graph::DistributionType::GEOMETRIC,
      graph::AtomicType::NATURAL,
      std::vector<uint>{c_prob});
  g.add_distribution(
      graph::DistributionType::HALF_CAUCHY,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{c_pos});
  g.add_distribution(
      graph::DistributionType::HALF_NORMAL,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{c_pos});
  g.add_distribution(
      graph::DistributionType::LOG_NORMAL,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{c_real1, c_pos1});
  uint d_normal = g.add_distribution(
      graph::DistributionType::NORMAL,
      graph::AtomicType::REAL,
      std::vector<uint>{c_real0, c_pos});
  g.add_distribution(
      graph::DistributionType::POISSON,
      graph::AtomicType::NATURAL,
      std::vector<uint>{c_pos});
  g.add_distribution(
      graph::DistributionType::STUDENT_T,
      graph::AtomicType::REAL,
      std::vector<uint>{c_pos, c_real0, c_pos});
  g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c3});

  // operators
  uint o_sample_bool = g.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{d_bernoulli});
  uint o_sample_real =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_normal});
  g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_binomial});
  uint o_sample_prob =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_beta});
  uint o_sample_pos =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_gamma});

  g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_bernoulli, c_natural_1, c_natural_2});
  g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_normal, c_natural_2, c_natural_2});
  g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_binomial, c_natural_2});
  g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_beta, c_natural_2, c_natural_1});
  g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_gamma, c_natural_2, c_natural_2});

  uint o_to_real = g.add_operator(
      graph::OperatorType::TO_REAL, std::vector<uint>{o_sample_pos});
  uint o_to_pos = g.add_operator(
      graph::OperatorType::TO_POS_REAL, std::vector<uint>{o_sample_prob});
  g.add_operator(
      graph::OperatorType::COMPLEMENT, std::vector<uint>{o_sample_prob});
  g.add_operator(graph::OperatorType::NEGATE, std::vector<uint>{c_real0});
  g.add_operator(graph::OperatorType::EXP, std::vector<uint>{c_real0});
  g.add_operator(graph::OperatorType::EXPM1, std::vector<uint>{o_sample_pos});
  uint o_log =
      g.add_operator(graph::OperatorType::LOG, std::vector<uint>{o_to_pos});
  uint o_log1pexp =
      g.add_operator(graph::OperatorType::LOG1PEXP, std::vector<uint>{o_log});
  g.add_operator(graph::OperatorType::LOG1MEXP, std::vector<uint>{c_neg});
  uint o_logsumexp = g.add_operator(
      graph::OperatorType::LOGSUMEXP,
      std::vector<uint>{c_real0, o_sample_real});
  g.add_operator(
      graph::OperatorType::MULTIPLY,
      std::vector<uint>{c_real0, o_sample_real, o_logsumexp});
  g.add_operator(
      graph::OperatorType::ADD,
      std::vector<uint>{c_real0, o_sample_real, o_to_real});
  g.add_operator(graph::OperatorType::PHI, std::vector<uint>{o_sample_real});
  g.add_operator(graph::OperatorType::LOGISTIC, std::vector<uint>{c_real0});
  g.add_operator(
      graph::OperatorType::IF_THEN_ELSE,
      std::vector<uint>{o_sample_bool, o_sample_pos, o_log1pexp});
  // factors
  g.add_factor(
      graph::FactorType::EXP_PRODUCT,
      std::vector<uint>{o_sample_real, o_sample_pos, o_to_pos});

  // copy and test
  graph::Graph g_copy(g);
  ASSERT_EQ(g.to_string(), g_copy.to_string());
  ASSERT_EQ(g.collect_statistics(), g_copy.collect_statistics());

  std::string expected_output("Graph Statistics Report\n");
  expected_output += "#######################\n";
  expected_output += "Number of nodes: 62\n";
  expected_output += "Number of edges: 71\n";
  expected_output += "Graph density: 0.02\n";
  expected_output += "Number of root nodes: 19\n";
  expected_output += "Number of terminal nodes: 39\n";
  expected_output += "Maximum no. of incoming edges into a node: 3\n";
  expected_output += "Maximum no. of outgoing edges from a node: 11\n";
  expected_output += "\n";
  expected_output += "Node statistics:\n";
  expected_output += "################\n";
  expected_output += "CONSTANT: 18\n";
  expected_output += "\tRoot nodes: 18\n";
  expected_output += "\tTerminal nodes: 9\n";
  expected_output += "\tConstant node statistics:\n";
  expected_output += "\t-------------------------\n";
  expected_output += "\t\tBOOLEAN and SCALAR: 1\n";
  expected_output += "\t\tBOOLEAN and BROADCAST_MATRIX: 1\n";
  expected_output += "\t\tPROBABILITY and SCALAR: 1\n";
  expected_output += "\t\tPROBABILITY and BROADCAST_MATRIX: 1\n";
  expected_output += "\t\tPROBABILITY and COL_SIMPLEX_MATRIX: 1\n";
  expected_output += "\t\tREAL and SCALAR: 3\n";
  expected_output += "\t\tREAL and BROADCAST_MATRIX: 1\n";
  expected_output += "\t\tPOS_REAL and SCALAR: 2\n";
  expected_output += "\t\tPOS_REAL and BROADCAST_MATRIX: 2\n";
  expected_output += "\t\tNATURAL and SCALAR: 3\n";
  expected_output += "\t\tNATURAL and BROADCAST_MATRIX: 1\n";
  expected_output += "\t\tNEG_REAL and SCALAR: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 18\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 9\n";
  expected_output += "\t\tNodes with 1 edges: 2\n";
  expected_output += "\t\tNodes with 2 edges: 3\n";
  expected_output += "\t\tNodes with 4 edges: 1\n";
  expected_output += "\t\tNodes with 8 edges: 1\n";
  expected_output += "\t\tNodes with 10 edges: 1\n";
  expected_output += "\t\tNodes with 11 edges: 1\n";
  expected_output += "\n";
  expected_output += "DISTRIBUTION: 18\n";
  expected_output += "\tRoot nodes: 1\n";
  expected_output += "\tTerminal nodes: 13\n";
  expected_output += "\tDistribution node statistics:\n";
  expected_output += "\t-----------------------------\n";
  expected_output += "\t\tTABULAR: 1\n";
  expected_output += "\t\tBERNOULLI: 1\n";
  expected_output += "\t\tBERNOULLI_NOISY_OR: 1\n";
  expected_output += "\t\tBETA: 1\n";
  expected_output += "\t\tBINOMIAL: 1\n";
  expected_output += "\t\tFLAT: 1\n";
  expected_output += "\t\tNORMAL: 1\n";
  expected_output += "\t\tLOG_NORMAL: 1\n";
  expected_output += "\t\tHALF_NORMAL: 1\n";
  expected_output += "\t\tHALF_CAUCHY: 1\n";
  expected_output += "\t\tSTUDENT_T: 1\n";
  expected_output += "\t\tBERNOULLI_LOGIT: 1\n";
  expected_output += "\t\tGAMMA: 1\n";
  expected_output += "\t\tBIMIXTURE: 1\n";
  expected_output += "\t\tCATEGORICAL: 1\n";
  expected_output += "\t\tPOISSON: 1\n";
  expected_output += "\t\tGEOMETRIC: 1\n";
  expected_output += "\t\tCAUCHY: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 1\n";
  expected_output += "\t\tNodes with 1 edges: 9\n";
  expected_output += "\t\tNodes with 2 edges: 6\n";
  expected_output += "\t\tNodes with 3 edges: 2\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 13\n";
  expected_output += "\t\tNodes with 2 edges: 4\n";
  expected_output += "\t\tNodes with 4 edges: 1\n";
  expected_output += "\n";
  expected_output += "OPERATOR: 25\n";
  expected_output += "\tTerminal nodes: 16\n";
  expected_output += "\tOperator node statistics:\n";
  expected_output += "\t-------------------------\n";
  expected_output += "\t\tSAMPLE: 5\n";
  expected_output += "\t\tIID_SAMPLE: 5\n";
  expected_output += "\t\tTO_REAL: 1\n";
  expected_output += "\t\tTO_POS_REAL: 1\n";
  expected_output += "\t\tCOMPLEMENT: 1\n";
  expected_output += "\t\tNEGATE: 1\n";
  expected_output += "\t\tEXP: 1\n";
  expected_output += "\t\tEXPM1: 1\n";
  expected_output += "\t\tMULTIPLY: 1\n";
  expected_output += "\t\tADD: 1\n";
  expected_output += "\t\tPHI: 1\n";
  expected_output += "\t\tLOGISTIC: 1\n";
  expected_output += "\t\tIF_THEN_ELSE: 1\n";
  expected_output += "\t\tLOG1PEXP: 1\n";
  expected_output += "\t\tLOGSUMEXP: 1\n";
  expected_output += "\t\tLOG: 1\n";
  expected_output += "\t\tLOG1MEXP: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 1 edges: 16\n";
  expected_output += "\t\tNodes with 2 edges: 2\n";
  expected_output += "\t\tNodes with 3 edges: 7\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 16\n";
  expected_output += "\t\tNodes with 1 edges: 5\n";
  expected_output += "\t\tNodes with 2 edges: 2\n";
  expected_output += "\t\tNodes with 4 edges: 1\n";
  expected_output += "\t\tNodes with 5 edges: 1\n";
  expected_output += "\n";
  expected_output += "FACTOR: 1\n";
  expected_output += "\tTerminal nodes: 1\n";
  expected_output += "\tFactor node statistics:\n";
  expected_output += "\t-----------------------\n";
  expected_output += "\t\tEXP_PRODUCT: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of incoming edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 3 edges: 1\n";
  expected_output += "\n";
  expected_output += "\t\tDistribution of outgoing edges:\n";
  expected_output += "\t\t-------------------------------\n";
  expected_output += "\t\tNodes with 0 edges: 1\n";
  expected_output += "\n";
  expected_output += "Edge statistics:\n";
  expected_output += "################\n";
  expected_output += "\tDistribution of incoming edges:\n";
  expected_output += "\t-------------------------------\n";
  expected_output += "\tNodes with 0 edges: 19\n";
  expected_output += "\tNodes with 1 edges: 25\n";
  expected_output += "\tNodes with 2 edges: 8\n";
  expected_output += "\tNodes with 3 edges: 10\n";
  expected_output += "\n";
  expected_output += "\tDistribution of outgoing edges:\n";
  expected_output += "\t-------------------------------\n";
  expected_output += "\tNodes with 0 edges: 39\n";
  expected_output += "\tNodes with 1 edges: 7\n";
  expected_output += "\tNodes with 2 edges: 9\n";
  expected_output += "\tNodes with 4 edges: 3\n";
  expected_output += "\tNodes with 5 edges: 1\n";
  expected_output += "\tNodes with 8 edges: 1\n";
  expected_output += "\tNodes with 10 edges: 1\n";
  expected_output += "\tNodes with 11 edges: 1\n";
  expected_output += "\n";

  ASSERT_EQ(expected_output, g.collect_statistics());
}

} // namespace graph
} // namespace beanmachine
