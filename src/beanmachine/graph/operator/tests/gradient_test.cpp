/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  torch::Tensor Xmat(3, 2);
  Xmat << 1.0, 0.5, 1.0, -1.5, 1.0, 2.1;
  for (uint i = 0; i < 3; ++i) {
    torch::Tensor x = Xmat.row(i);
    uint x_i = g.add_constant_real_matrix(x);
    uint mu_i = g.add_operator(
        OperatorType::MATRIX_MULTIPLY, std::vector<uint>{x_i, betas});
    uint dist_i = g.add_distribution(
        DistributionType::NORMAL,
        AtomicType::REAL,
        std::vector<uint>{mu_i, sd});
    uint y_i = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist_i});
    g.observe(y_i, 0.5);
  }
  torch::Tensor betas_mat(2, 1);
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

TEST(testgradient, backward_unaryops) {
  Graph g;
  auto flat_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto prob =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_dist});
  auto c_prob =
      g.add_operator(OperatorType::COMPLEMENT, std::vector<uint>{prob});
  auto pos_real =
      g.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>{c_prob});
  auto neg_real =
      g.add_operator(OperatorType::NEGATE, std::vector<uint>{pos_real});
  auto real =
      g.add_operator(OperatorType::TO_REAL, std::vector<uint>{neg_real});
  auto exp_ = g.add_operator(OperatorType::EXP, std::vector<uint>{pos_real});
  auto expm1 = g.add_operator(OperatorType::EXPM1, std::vector<uint>{real});
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{expm1, exp_});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  g.observe(prob, 0.67);
  g.observe(y, 0.14);
  // To verify the grad1 results with pyTorch:
  // x = tensor([0.67], requires_grad=True)
  // mu = (x - tensor([1.0])).expm1()
  // sd = (tensor([1.0]) - x).exp()
  // log_p = dist.Normal(mu, sd).log_prob((tensor(0.14)))
  // torch.autograd.grad(log_p, x) -> 1.0648
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_double, 1.0648, 1e-3);
}

TEST(testgradient, backward_matrix_index) {
  Graph g;

  torch::Tensor m1(3, 1);
  m1 << 2.0, 0.5, 3.0;
  uint cm1 = g.add_constant_pos_matrix(m1);
  uint one = g.add_constant((natural_t)1);
  uint half = g.add_constant_probability(0.5);

  uint diri_dist = g.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      std::vector<uint>{cm1});
  uint diri_sample = g.add_operator(OperatorType::SAMPLE, {diri_dist});

  torch::Tensor obs(3, 1);
  obs << 0.4, 0.1, 0.5;
  g.observe(diri_sample, obs);

  uint diri_index = g.add_operator(OperatorType::INDEX, {diri_sample, one});
  uint half_prob = g.add_operator(OperatorType::MULTIPLY, {diri_index, half});

  uint bernoulli_dist = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half_prob});

  uint bernoulli_sample =
      g.add_operator(OperatorType::SAMPLE, {bernoulli_dist});
  g.observe(bernoulli_sample, true);

  // PyTorch verification
  // diri = dist.Dirichlet(tensor([2.0, 0.5, 3.0]))
  // diri_sample = tensor([0.4, 0.1, 0.5], requires_grad=True)
  // half_prob = diri_sample[1] * 0.5
  // bern = dist.Bernoulli(half_prob)
  // bern_sample = tensor(1.0, requires_grad=True)
  // log_prob = (bern.log_prob(bern_sample).sum()
  //    + diri.log_prob(diri_sample).sum())
  // grad(log_prob, diri_sample)
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_matrix(0), 2.5, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix(1), 5.0, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix(2), 4.0, 1e-3);
}

TEST(testgradient, backward_column_index) {
  Graph g;

  torch::Tensor m1(3, 2);
  m1 << 2.0, 1.0, 0.5, 3.0, 3.0, 2.0;
  uint cm1 = g.add_constant_pos_matrix(m1);
  uint zero = g.add_constant((natural_t)0);

  uint first_column = g.add_operator(OperatorType::COLUMN_INDEX, {cm1, zero});

  uint one = g.add_constant((natural_t)1);
  uint half = g.add_constant_probability(0.5);

  uint diri_dist = g.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      std::vector<uint>{first_column});
  uint diri_sample = g.add_operator(OperatorType::SAMPLE, {diri_dist});

  torch::Tensor obs(3, 1);
  obs << 0.4, 0.1, 0.5;
  g.observe(diri_sample, obs);

  uint diri_index = g.add_operator(OperatorType::INDEX, {diri_sample, one});
  uint half_prob = g.add_operator(OperatorType::MULTIPLY, {diri_index, half});

  uint bernoulli_dist = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half_prob});

  uint bernoulli_sample =
      g.add_operator(OperatorType::SAMPLE, {bernoulli_dist});
  g.observe(bernoulli_sample, true);

  // PyTorch verification
  // diri = dist.Dirichlet(tensor([2.0, 0.5, 3.0]))
  // diri_sample = tensor([0.4, 0.1, 0.5], requires_grad=True)
  // half_prob = diri_sample[1] * 0.5
  // bern = dist.Bernoulli(half_prob)
  // bern_sample = tensor(1.0, requires_grad=True)
  // log_prob = (bern.log_prob(bern_sample).sum()
  //    + diri.log_prob(diri_sample).sum())
  // grad(log_prob, diri_sample)
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_matrix(0), 2.5, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix(1), 5.0, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix(2), 4.0, 1e-3);
}

TEST(testgradient, forward_to_matrix) {
  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint five = g.add_constant_pos_real(5.0);
  uint nat_one = g.add_constant((natural_t)1);
  uint nat_two = g.add_constant((natural_t)2);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint stu_dist = g.add_distribution(
      DistributionType::STUDENT_T, AtomicType::REAL, {five, zero, two});
  uint stu_sample = g.add_operator(OperatorType::SAMPLE, {stu_dist});
  g.observe(stu_sample, 0.1);

  uint sample_matrix = g.add_operator(
      OperatorType::TO_MATRIX, {nat_two, nat_one, norm_sample, stu_sample});

  uint zero_nat = g.add_constant((natural_t)0);
  uint sample_index =
      g.add_operator(OperatorType::INDEX, {sample_matrix, zero_nat});
  uint new_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample_index, one});
  uint new_norm_sample = g.add_operator(OperatorType::SAMPLE, {new_norm_dist});
  g.observe(new_norm_sample, 0.6);

  /* p ~ Normal(0, 1)
     p2 ~ Normal(p, 1)
     p2 = 0.6
     posterior: Normal(0.5 * 0.6, 0.5) = N(0.3, 0.5)
     grad  w.r.t. m : (x - m) / s^2 = 0.6
     grad2 w.r.t. m : -1 / s^2 = -2
  */
  double grad1;
  double grad2;
  g.gradient_log_prob(norm_sample, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6, 1e-3);
  EXPECT_NEAR(grad2, -2, 1e-3);
}

TEST(testgradient, backward_to_matrix) {
  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint five = g.add_constant_pos_real(5.0);
  uint nat_two = g.add_constant((natural_t)2);
  uint nat_one = g.add_constant((natural_t)1);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});
  g.observe(norm_sample, 0.5);

  uint stu_dist = g.add_distribution(
      DistributionType::STUDENT_T, AtomicType::REAL, {five, zero, two});
  uint stu_sample = g.add_operator(OperatorType::SAMPLE, {stu_dist});
  g.observe(stu_sample, 0.1);

  uint sample_matrix = g.add_operator(
      OperatorType::TO_MATRIX, {nat_two, nat_one, norm_sample, stu_sample});

  uint zero_nat = g.add_constant((natural_t)0);
  uint sample_index =
      g.add_operator(OperatorType::INDEX, {sample_matrix, zero_nat});
  uint new_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample_index, one});
  uint new_norm_sample = g.add_operator(OperatorType::SAMPLE, {new_norm_dist});
  g.observe(new_norm_sample, 0.6);

  // PyTorch Verification
  // norm_dist = dist.Normal(0., 1.)
  // stu_dist = dist.StudentT(5, 0., 2.)
  // norm_sample = tensor(0.5, requires_grad=True)
  // stu_sample = tensor(0.1, requires_grad=True)
  // new_norm = dist.Normal(norm_sample, 1.)
  // new_sample = tensor(0.6, requires_grad=True)
  // log_prob = (norm_dist.log_prob(norm_sample) + stu_dist.log_prob(stu_sample)
  //    + new_norm.log_prob(new_sample))
  // grad(log_prob, norm_sample) -> -0.4
  // grad(log_prob, stu_sample) -> -0.03
  // grad(log_prob, new_sample) -> -0.1

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR(grad1[0]->_double, -0.4, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, -0.03, 1e-3);
  EXPECT_NEAR(grad1[2]->_double, -0.1, 1e-3);
}

TEST(testgradient, forward_matrix_scale) {
  // This is a minimal test following forward_to_matrix above
  // as a template. The immediate purpose is to test that we
  // have the types (including matrix dimensions) right in this
  // diff. TODO[Walid]: Add test cases when scalar has non-zero
  // first and second gradients.

  // We will reuse essentially the same test for to_matrix
  // (see forward_to_matrix above), adding a simple use
  // of matrix scale in between operators to check that they
  // are providing values of the right type/shape.
  // To recapitulate, that model is:
  /* p ~ Normal(0, 1)
     p2 ~ Normal(p, 1)
     p2 = 0.6
     posterior: Normal(0.5 * 0.6, 0.5) = N(0.3, 0.5)
     grad  w.r.t. m : (x - m) / s^2 = 0.6
     grad2 w.r.t. m : -1 / s^2 = -2
  */
  // That code that follows will also be essentially
  // the same as that test, with the exception of the
  // new variable sample_scaled_matrix which scales
  // sample_matrix by 1, and we also use that new
  // variable in place of the old one in the rest of
  // computation.

  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint five = g.add_constant_pos_real(5.0);
  uint real_one = g.add_constant(1.0);
  uint nat_one = g.add_constant((natural_t)1);
  uint nat_two = g.add_constant((natural_t)2);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint stu_dist = g.add_distribution(
      DistributionType::STUDENT_T, AtomicType::REAL, {five, zero, two});
  uint stu_sample = g.add_operator(OperatorType::SAMPLE, {stu_dist});
  g.observe(stu_sample, 0.1);

  uint sample_matrix = g.add_operator(
      OperatorType::TO_MATRIX, {nat_two, nat_one, norm_sample, stu_sample});

  uint sample_scaled_matrix =
      g.add_operator(OperatorType::MATRIX_SCALE, {real_one, sample_matrix});

  uint zero_nat = g.add_constant((natural_t)0);
  uint sample_index =
      g.add_operator(OperatorType::INDEX, {sample_scaled_matrix, zero_nat});
  uint new_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample_index, one});
  uint new_norm_sample = g.add_operator(OperatorType::SAMPLE, {new_norm_dist});
  g.observe(new_norm_sample, 0.6);

  double grad1;
  double grad2;
  g.gradient_log_prob(norm_sample, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6, 1e-3);
  EXPECT_NEAR(grad2, -2, 1e-3);
}

TEST(testgradient, forward_broadcast_add) {
  Graph g;
  auto zero = g.add_constant(0.0);
  auto one = g.add_constant_pos_real(1.0);
  auto norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  auto norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  auto nat_zero = g.add_constant((natural_t)0);
  auto nat_one = g.add_constant((natural_t)1);
  torch::Tensor m(2, 1);
  m << 1.0, 2.0;
  auto matrix = g.add_constant_real_matrix(m);
  auto sum_matrix =
      g.add_operator(OperatorType::BROADCAST_ADD, {norm_sample, matrix});
  auto first_entry =
      g.add_operator(OperatorType::INDEX, {sum_matrix, nat_zero});
  auto second_entry =
      g.add_operator(OperatorType::INDEX, {sum_matrix, nat_one});
  auto mu = g.add_operator(OperatorType::MULTIPLY, {first_entry, second_entry});
  auto new_norm_dist =
      g.add_distribution(DistributionType::NORMAL, AtomicType::REAL, {mu, one});
  auto new_norm_sample = g.add_operator(OperatorType::SAMPLE, {new_norm_dist});
  g.observe(norm_sample, -0.5);
  g.observe(new_norm_sample, 1.0);
  /*
    PyTorch verification:
    def f(x):
        tmp = x + torch.tensor([1.0, 2.0])
        mu = tmp[0] * tmp[1]
        return dist.Normal(0.0, 1.0).log_prob(x) + \
                dist.Normal(mu, 1.0).log_prob(torch.tensor(1.0))
    grad1 = torch.autograd.functional.jacobian(f, torch.tensor(-0.5)) -> 1.0
    grad2 = torch.autograd.functional.hessian(f, torch.tensor(-0.5)) -> -4.5
  */
  double grad1;
  double grad2;
  g.gradient_log_prob(norm_sample, grad1, grad2);
  EXPECT_NEAR(grad1, 1.0, 1e-3);
  EXPECT_NEAR(grad2, -4.5, 1e-3);
}

TEST(testgradient, backward_broadcast_add) {
  Graph g;
  auto zero = g.add_constant(0.0);
  auto one = g.add_constant_pos_real(1.0);
  auto two = g.add_constant_pos_real(2.0);
  auto five = g.add_constant_pos_real(5.0);

  auto stu_dist = g.add_distribution(
      DistributionType::STUDENT_T, AtomicType::REAL, {five, zero, two});
  auto stu_sample = g.add_operator(OperatorType::SAMPLE, {stu_dist});
  g.observe(stu_sample, 0.5);

  torch::Tensor m(2, 1);
  m << -1.0, 3.0;
  auto matrix = g.add_constant_real_matrix(m);
  auto sum_matrix =
      g.add_operator(OperatorType::BROADCAST_ADD, {stu_sample, matrix});
  auto nat_zero = g.add_constant((natural_t)0);
  auto nat_one = g.add_constant((natural_t)1);
  auto first_entry =
      g.add_operator(OperatorType::INDEX, {sum_matrix, nat_zero});
  auto second_entry =
      g.add_operator(OperatorType::INDEX, {sum_matrix, nat_one});
  auto mu = g.add_operator(OperatorType::ADD, {first_entry, second_entry});
  auto norm_dist =
      g.add_distribution(DistributionType::NORMAL, AtomicType::REAL, {mu, one});
  auto norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});
  g.observe(norm_sample, 0.0);
  /*
    PyTorch verification:
    stu_dist = dist.StudentT(5.0, 0.0, 2.0)
    stu_sample = torch.tensor(0.5, requires_grad=True)
    sum_matrix = stu_sample + torch.tensor([-1.0, 3.0])
    mu = sum_matrix[0] + sum_matrix[1]
    norm_dist = dist.Normal(mu, 1.0)
    norm_sample = torch.tensor(0.0, requires_grad=True)
    log_prob = stu_dist.log_prob(stu_sample) + norm_dist.log_prob(norm_sample)

    torch.autograd.grad(log_prob, stu_sample) -> -6.1481
    torch.autograd.grad(log_prob, norm_sample) -> 3.0
  */
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_double, -6.1481, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, 3.0, 1e-3);
}
