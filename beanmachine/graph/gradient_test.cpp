// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testgradient, operators) {
  Graph g;
  AtomicValue value;
  double grad1;
  double grad2;
  // test operators on real numbers
  auto a = g.add_constant(3.0);
  auto b = g.add_constant(10.0);
  auto c = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({a, b}));
  auto d = g.add_operator(OperatorType::ADD, std::vector<uint>({a, a, c}));
  auto e = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({a, b, c, d}));
  auto f = g.add_operator(OperatorType::NEGATE, std::vector<uint>({e}));
  // c = 10a, where a=3. Therefore value=30, grad1=10, and grad2=0
  g.eval_and_grad(c, a, 12351, value, grad1, grad2);
  EXPECT_NEAR(value._double, 30, 0.001);
  EXPECT_NEAR(grad1, 10, 1e-6);
  EXPECT_NEAR(grad2, 0, 1e-6);
  // b=10, c=10a, d=12a, e= 1200a^3 f=-1200a^3
  // Therefore value=-32400, grad1=-3600a^2=-32400, and grad2=-7200a = -21600
  g.eval_and_grad(f, a, 12351, value, grad1, grad2);
  EXPECT_NEAR(value._double, -32400, 0.001);
  EXPECT_NEAR(grad1, -32400, 1e-3);
  EXPECT_NEAR(grad2, -21600, 1e-3);
  // test operators on probabilities
  auto h = g.add_constant_probability(0.3);
  auto i = g.add_operator(OperatorType::COMPLEMENT, std::vector<uint>({h}));
  auto j = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({h, i}));
  auto k = g.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>({j}));
  // k = h (1 -h); h=.3 => value = .21, grad1 = 1 - 2h = 0.4, and grad2 = -2
  g.eval_and_grad(k, h, 12351, value, grad1, grad2);
  EXPECT_NEAR(value._double, 0.21, 1e-6);
  EXPECT_NEAR(grad1, 0.4, 1e-6);
  EXPECT_NEAR(grad2, -2, 1e-6);
}

TEST(testgradient, beta_bernoulli) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint beta = g.add_distribution(
    DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint bernoulli = g.add_distribution(
    DistributionType::BERNOULLI, AtomicType::BOOLEAN, std::vector<uint>({prob}));
  uint var1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var3 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  double prob_val = 0.4;
  g.observe(prob, prob_val);
  g.observe(var1, true);
  g.observe(var2, true);
  g.observe(var3, false);
  // Note: the posterior is p ~ Beta(7, 4). Therefore log posterior is 6 log(p) + 3 log(1-p)
  // grad1 should be 6/p - 3/(1-p). And grad2 should be -6/p^2 - 3/(1-p)^2
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(prob, grad1, grad2);
  EXPECT_NEAR(grad1, 6 / prob_val - 3 / (1-prob_val), 1e-3);
  EXPECT_NEAR(grad2, -6 / (prob_val * prob_val) - 3 / ((1-prob_val)*(1-prob_val)), 1e-3);
  // Now consider a slightly more complicated case where first and second gradients are used
  uint prob2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint prob2sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({prob2, prob2}));
  uint bernoulli2 = g.add_distribution(
    DistributionType::BERNOULLI, AtomicType::BOOLEAN, std::vector<uint>({prob2sq}));
  uint var2_1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli2}));
  uint var2_2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli2}));
  uint var2_3 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli2}));
  g.observe(prob2, prob_val);
  g.observe(var2_1, true);
  g.observe(var2_2, true);
  g.observe(var2_3, false);
  grad1 = grad2 = 0;
  g.gradient_log_prob(prob2, grad1, grad2);
  // f(p) = 4 log(p) + 2 log(1-p) + 2 log(p^2) + log(1 - p^2)
  // simplifying: f(p) = 8 log(p) + 3 log(1-p) + log(1+p)
  // f'(p) = 8/p - 3/(1-p) + 1/(1+p)
  // f''(p) = -8/p^2 -3/(1-p)^2 -1/(1+p)^2
  EXPECT_NEAR(grad1, 8/prob_val - 3/(1-prob_val) + 1/(1+prob_val), 1e-3);
  EXPECT_NEAR(grad2,
    -8/(prob_val*prob_val) - 3/((1-prob_val)*(1-prob_val)) - 1/((1+prob_val)*(1+prob_val)), 1e-3);
}

TEST(testgradient, beta_bernoulli_noisy_or) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint beta = g.add_distribution(
    DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint probsq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({prob, prob}));
  uint probsq_pos_real = g.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>({probsq}));
  uint bernoulli = g.add_distribution(
    DistributionType::BERNOULLI_NOISY_OR, AtomicType::BOOLEAN, std::vector<uint>({probsq_pos_real}));
  uint var1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var3 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  double p = 0.4;
  g.observe(prob, p);
  g.observe(var1, true);
  g.observe(var2, true);
  g.observe(var3, false);
  // f(p)   =  4 log(p) + 2 log(1-p) + 2 log(1 - exp(-p^2)) - p^2
  // f'(p)  =  4/p - 2/(1-p) + 4p exp(-p^2) / (1 - exp(-p^2)) - 2p
  //        =  4/p - 2/(1-p) + 4p / (1 - exp(-p^2)) - 6p
  // f''(p) =  -4/p^2 - 2/(1-p)^2 + 4 / (1 - exp(-p^2)) - 8p^2 exp(-p^2) / (1 - exp(-p^2))^2 -6
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(prob, grad1, grad2);
  EXPECT_NEAR(grad1, 4/p - 2/(1-p) + 4*p / (1 - std::exp(-p*p)) - 6 * p, 1e-3);
  EXPECT_NEAR(grad2,
    -4/(p*p) - 2/((1-p)*(1-p)) + 4 / (1 - std::exp(-p*p))
    - 8 * p * p * exp(-p*p) / ((1 - exp(-p*p))*(1 - exp(-p*p))) -6, 1e-3);
}

TEST(testgradient, beta_binomial) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint n = g.add_constant((natural_t) 3);
  uint beta = g.add_distribution(
    DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint binomial = g.add_distribution(
    DistributionType::BINOMIAL, AtomicType::NATURAL, std::vector<uint>({n, prob}));
  uint k = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({binomial}));
  double prob_val = 0.4;
  g.observe(prob, prob_val);
  g.observe(k, (natural_t) 2);
  // Note: the posterior is p ~ Beta(7, 4). Therefore log posterior is 6 log(p) + 3 log(1-p)
  // grad1 should be 6/p - 3/(1-p). And grad2 should be -6/p^2 - 3/(1-p)^2
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(prob, grad1, grad2);
  EXPECT_NEAR(grad1, 6 / prob_val - 3 / (1-prob_val), 1e-3);
  EXPECT_NEAR(grad2, -6 / (prob_val * prob_val) - 3 / ((1-prob_val)*(1-prob_val)), 1e-3);
  // Now consider a slightly more complicated case where first and second gradients are used
  uint prob2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint prob2sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({prob2, prob2}));
  uint binomial2 = g.add_distribution(
    DistributionType::BINOMIAL, AtomicType::NATURAL, std::vector<uint>({n, prob2sq}));
  uint k2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({binomial2}));
  g.observe(prob2, prob_val);
  g.observe(k2, (natural_t) 2);
  grad1 = grad2 = 0;
  g.gradient_log_prob(prob2, grad1, grad2);
  // f(p) = 4 log(p) + 2 log(1-p) + 2 log(p^2) + log(1 - p^2)
  // simplifying: f(p) = 8 log(p) + 3 log(1-p) + log(1+p)
  // f'(p) = 8/p - 3/(1-p) + 1/(1+p)
  // f''(p) = -8/p^2 -3/(1-p)^2 -1/(1+p)^2
  EXPECT_NEAR(grad1, 8/prob_val - 3/(1-prob_val) + 1/(1+prob_val), 1e-3);
  EXPECT_NEAR(grad2,
    -8/(prob_val*prob_val) - 3/((1-prob_val)*(1-prob_val)) - 1/((1+prob_val)*(1+prob_val)), 1e-3);
}
