// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testgradient, operators) {
  Graph g;
  NodeValue value;
  double grad1;
  double grad2;
  // test operators on real numbers
  auto a = g.add_constant(3.0);
  auto b = g.add_constant(10.0);
  auto c = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({a, b}));
  auto d = g.add_operator(OperatorType::ADD, std::vector<uint>({a, a, c}));
  auto e =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({a, b, c, d}));
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
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI,
      AtomicType::BOOLEAN,
      std::vector<uint>({prob}));
  uint var1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  double prob_val = 0.4;
  g.observe(prob, prob_val);
  g.observe(var1, true);
  g.observe(var2, true);
  g.observe(var3, false);
  // Note: the posterior is p ~ Beta(7, 4). Therefore log posterior is 6 log(p)
  // + 3 log(1-p) grad1 should be 6/p - 3/(1-p). And grad2 should be -6/p^2 -
  // 3/(1-p)^2
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(prob, grad1, grad2);
  EXPECT_NEAR(grad1, 6 / prob_val - 3 / (1 - prob_val), 1e-3);
  EXPECT_NEAR(
      grad2,
      -6 / (prob_val * prob_val) - 3 / ((1 - prob_val) * (1 - prob_val)),
      1e-3);
  // Now consider a slightly more complicated case where first and second
  // gradients are used
  uint prob2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint prob2sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({prob2, prob2}));
  uint bernoulli2 = g.add_distribution(
      DistributionType::BERNOULLI,
      AtomicType::BOOLEAN,
      std::vector<uint>({prob2sq}));
  uint var2_1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli2}));
  uint var2_2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli2}));
  uint var2_3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli2}));
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
  EXPECT_NEAR(
      grad1, 8 / prob_val - 3 / (1 - prob_val) + 1 / (1 + prob_val), 1e-3);
  EXPECT_NEAR(
      grad2,
      -8 / (prob_val * prob_val) - 3 / ((1 - prob_val) * (1 - prob_val)) -
          1 / ((1 + prob_val) * (1 + prob_val)),
      1e-3);
}

TEST(testgradient, beta_bernoulli_noisy_or) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint beta = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint probsq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({prob, prob}));
  uint probsq_pos_real =
      g.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>({probsq}));
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI_NOISY_OR,
      AtomicType::BOOLEAN,
      std::vector<uint>({probsq_pos_real}));
  uint var1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  uint var3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({bernoulli}));
  double p = 0.4;
  g.observe(prob, p);
  g.observe(var1, true);
  g.observe(var2, true);
  g.observe(var3, false);
  // f(p)   =  4 log(p) + 2 log(1-p) + 2 log(1 - exp(-p^2)) - p^2
  // f'(p)  =  4/p - 2/(1-p) + 4p exp(-p^2) / (1 - exp(-p^2)) - 2p
  //        =  4/p - 2/(1-p) + 4p / (1 - exp(-p^2)) - 6p
  // f''(p) =  -4/p^2 - 2/(1-p)^2 + 4 / (1 - exp(-p^2)) - 8p^2 exp(-p^2) / (1 -
  // exp(-p^2))^2 -6
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(prob, grad1, grad2);
  EXPECT_NEAR(
      grad1,
      4 / p - 2 / (1 - p) + 4 * p / (1 - std::exp(-p * p)) - 6 * p,
      1e-3);
  EXPECT_NEAR(
      grad2,
      -4 / (p * p) - 2 / ((1 - p) * (1 - p)) + 4 / (1 - std::exp(-p * p)) -
          8 * p * p * exp(-p * p) / ((1 - exp(-p * p)) * (1 - exp(-p * p))) - 6,
      1e-3);
}

TEST(testgradient, beta_binomial) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint n = g.add_constant((natural_t)3);
  uint beta = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint binomial = g.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>({n, prob}));
  uint k = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({binomial}));
  double prob_val = 0.4;
  g.observe(prob, prob_val);
  g.observe(k, (natural_t)2);
  // Note: the posterior is p ~ Beta(7, 4). Therefore log posterior is 6 log(p)
  // + 3 log(1-p) grad1 should be 6/p - 3/(1-p). And grad2 should be -6/p^2 -
  // 3/(1-p)^2
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(prob, grad1, grad2);
  EXPECT_NEAR(grad1, 6 / prob_val - 3 / (1 - prob_val), 1e-3);
  EXPECT_NEAR(
      grad2,
      -6 / (prob_val * prob_val) - 3 / ((1 - prob_val) * (1 - prob_val)),
      1e-3);
  // Now consider a slightly more complicated case where first and second
  // gradients are used
  uint prob2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint prob2sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>({prob2, prob2}));
  uint binomial2 = g.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>({n, prob2sq}));
  uint k2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({binomial2}));
  g.observe(prob2, prob_val);
  g.observe(k2, (natural_t)2);
  grad1 = grad2 = 0;
  g.gradient_log_prob(prob2, grad1, grad2);
  // f(p) = 4 log(p) + 2 log(1-p) + 2 log(p^2) + log(1 - p^2)
  // simplifying: f(p) = 8 log(p) + 3 log(1-p) + log(1+p)
  // f'(p) = 8/p - 3/(1-p) + 1/(1+p)
  // f''(p) = -8/p^2 -3/(1-p)^2 -1/(1+p)^2
  EXPECT_NEAR(
      grad1, 8 / prob_val - 3 / (1 - prob_val) + 1 / (1 + prob_val), 1e-3);
  EXPECT_NEAR(
      grad2,
      -8 / (prob_val * prob_val) - 3 / ((1 - prob_val) * (1 - prob_val)) -
          1 / ((1 + prob_val) * (1 + prob_val)),
      1e-3);
}

TEST(testgradient, backward_scalar_linearmodel) {
  // constant: x_i, for i in {0, 1, 2}
  // prior: coeff_x, intercept ~ Normal(0, 1)
  // likelihood: y_i ~ Normal(x_i * coeff_x + intercept, 1)
  Graph g;
  uint zero = g.add_constant(0.0);
  uint pos_one = g.add_constant_pos_real(1.0);

  uint normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos_one});
  uint intercept =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  uint coeff_x =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});

  std::vector<double> X{0.5, -1.5, 2.1};
  for (double x : X) {
    uint x_i = g.add_constant(x);
    uint xb_i =
        g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x_i, coeff_x});
    uint mu_i =
        g.add_operator(OperatorType::ADD, std::vector<uint>{xb_i, intercept});
    uint dist_i = g.add_distribution(
        DistributionType::NORMAL,
        AtomicType::REAL,
        std::vector<uint>{mu_i, pos_one});
    uint y_i = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist_i});
    g.observe(y_i, 0.5);
  }
  g.observe(intercept, 0.1);
  g.observe(coeff_x, 0.2);

  // To verify the grad1 results with pyTorch:
  // b0 = tensor([0.1], requires_grad=True)
  // b1 = tensor([0.2], requires_grad=True)
  // x = tensor([0.5, -1.5, 2.1])
  // y = 0.5
  // log_p = (
  //     dist.Normal(x * b1 + b0, tensor(1.0)).log_prob(tensor([y, y, y])).sum()
  //     + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(b0)
  //     + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(b1)
  // )
  // torch.autograd.grad(log_p, b0) -> 0.8800
  // torch.autograd.grad(log_p, b1) -> -1.1420
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 5);
  EXPECT_NEAR(grad1[0]->_double, 0.8800, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, -1.1420, 1e-3);
}

TEST(testgradient, backward_vector_linearmodel) {
  // constant matrix(3 by 2): X
  // prior vector(2 by 1): betas ~ iid Normal(0, 1)
  // likelihood: y_i ~ Normal((X @ betas)[i], 1)
  Graph g;
  uint zero = g.add_constant(0.0);
  uint pos_one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant((natural_t)2);

  uint normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos_one});
  uint hc_dist = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos_one});
  uint betas = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two});
  uint sd = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{hc_dist});
  Eigen::MatrixXd Xmat(3, 2);
  Xmat << 1.0, 0.5, 1.0, -1.5, 1.0, 2.1;
  for (uint i = 0; i < 3; ++i) {
    Eigen::MatrixXd x = Xmat.row(i);
    uint x_i = g.add_constant_matrix(x);
    uint mu_i = g.add_operator(
        OperatorType::MATRIX_MULTIPLY, std::vector<uint>{x_i, betas});
    uint dist_i = g.add_distribution(
        DistributionType::NORMAL,
        AtomicType::REAL,
        std::vector<uint>{mu_i, sd});
    uint y_i = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist_i});
    g.observe(y_i, 0.5);
  }
  Eigen::MatrixXd betas_mat(2, 1);
  betas_mat << 0.1, 0.2;
  g.observe(betas, betas_mat);
  g.observe(sd, 1.4);

  // To verify the grad1 results with pyTorch:
  // betas = tensor([[0.1], [0.2]], requires_grad=True)
  // sd = tensor(1.4, requires_grad=True)
  // X = tensor([[1.0, 0.5], [1.0, -1.5], [1.0, 2.1]])
  // y = 0.5
  // Xb = torch.mm(X, betas)
  // log_p = (
  //     dist.Normal(Xb, sd).log_prob(tensor([[y], [y], [y]])).sum()
  //     + dist.Normal(
  //       tensor([[0.0], [0.0]]), tensor(1.0)).log_prob(betas).sum()
  //     + dist.HalfCauchy(tensor(1.0)).log_prob(sd)
  // )
  // torch.autograd.grad(log_p, betas) # -> 0.4000, -0.6806
  // torch.autograd.grad(log_p, sd) # -> -2.8773
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 5);
  EXPECT_NEAR(grad1[0]->_matrix.coeff(0), 0.4000, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix.coeff(1), -0.6806, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, -2.8773, 1e-3);
}
