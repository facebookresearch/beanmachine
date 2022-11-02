/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <beanmachine/graph/graph.h>
#include <beanmachine/graph/operator/linalgop.h>
#include <cstdlib>

using namespace beanmachine::graph;

// TODO: These test helpers should be moved into a test utility header.
bool MatrixEquality(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs) {
  return lhs.isApprox(rhs, 1e-4);
}

#define EXPECT_NEAR_MATRIX(lhs, rhs) ASSERT_PRED2(MatrixEquality, lhs, rhs)

#define EXPECT_NEAR_MATRIX_EPS(lhs, rhs, eps) \
  ASSERT_PRED2(                               \
      [=](auto& lhs, auto& rhs) { return lhs.isApprox(rhs, eps); }, lhs, rhs)

TEST(testgradient, operators) {
  Graph g;
  NodeValue value;
  double grad1;
  double grad2;
  // test operators on real numbers
  auto a = g.add_constant_real(3.0);
  auto b = g.add_constant_real(10.0);
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
  uint n = g.add_constant_natural(3);
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
  uint zero = g.add_constant_real(0.0);
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
    uint x_i = g.add_constant_real(x);
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
  EXPECT_NEAR((*grad1[0]), 0.8800, 1e-3);
  EXPECT_NEAR((*grad1[1]), -1.1420, 1e-3);
}

TEST(testgradient, backward_vector_linearmodel) {
  // constant matrix(3 by 2): X
  // prior vector(2 by 1): betas ~ iid Normal(0, 1)
  // likelihood: y_i ~ Normal((X @ betas)[i], 1)
  Graph g;
  uint zero = g.add_constant_real(0.0);
  uint pos_one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_natural(2);

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
  EXPECT_NEAR(grad1[0]->coeff(0), 0.4000, 1e-3);
  EXPECT_NEAR(grad1[0]->coeff(1), -0.6806, 1e-3);
  EXPECT_NEAR((*grad1[1]), -2.8773, 1e-3);
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
  EXPECT_NEAR((*grad1[0]), 1.0648, 1e-3);
}

TEST(testgradient, backward_matrix_index) {
  Graph g;

  Eigen::MatrixXd m1(3, 1);
  m1 << 2.0, 0.5, 3.0;
  uint cm1 = g.add_constant_pos_matrix(m1);
  uint one = g.add_constant_natural(1);
  uint half = g.add_constant_probability(0.5);

  uint diri_dist = g.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      std::vector<uint>{cm1});
  uint diri_sample = g.add_operator(OperatorType::SAMPLE, {diri_dist});

  Eigen::MatrixXd obs(3, 1);
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
  EXPECT_NEAR((*grad1[0])(0), 2.5, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), 5.0, 1e-3);
  EXPECT_NEAR((*grad1[0])(2), 4.0, 1e-3);
}

TEST(testgradient, backward_column_index) {
  Graph g;

  Eigen::MatrixXd m1(3, 2);
  m1 << 2.0, 1.0, 0.5, 3.0, 3.0, 2.0;
  uint cm1 = g.add_constant_pos_matrix(m1);
  uint zero = g.add_constant_natural(0);

  uint first_column = g.add_operator(OperatorType::COLUMN_INDEX, {cm1, zero});

  uint one = g.add_constant_natural(1);
  uint half = g.add_constant_probability(0.5);

  uint diri_dist = g.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      std::vector<uint>{first_column});
  uint diri_sample = g.add_operator(OperatorType::SAMPLE, {diri_dist});

  Eigen::MatrixXd obs(3, 1);
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
  EXPECT_NEAR((*grad1[0])(0), 2.5, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), 5.0, 1e-3);
  EXPECT_NEAR((*grad1[0])(2), 4.0, 1e-3);
}

TEST(testgradient, forward_to_matrix) {
  Graph g;
  uint zero = g.add_constant_real(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint five = g.add_constant_pos_real(5.0);
  uint nat_one = g.add_constant_natural(1);
  uint nat_two = g.add_constant_natural(2);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint stu_dist = g.add_distribution(
      DistributionType::STUDENT_T, AtomicType::REAL, {five, zero, two});
  uint stu_sample = g.add_operator(OperatorType::SAMPLE, {stu_dist});
  g.observe(stu_sample, 0.1);

  uint sample_matrix = g.add_operator(
      OperatorType::TO_MATRIX, {nat_two, nat_one, norm_sample, stu_sample});

  uint zero_nat = g.add_constant_natural(0);
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
  double grad1 = 0.0;
  double grad2 = 0.0;
  g.gradient_log_prob(norm_sample, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6, 1e-3);
  EXPECT_NEAR(grad2, -2, 1e-3);
}

TEST(testgradient, backward_to_matrix) {
  Graph g;
  uint zero = g.add_constant_real(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint five = g.add_constant_pos_real(5.0);
  uint nat_two = g.add_constant_natural(2);
  uint nat_one = g.add_constant_natural(1);

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

  uint zero_nat = g.add_constant_natural(0);
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
  EXPECT_NEAR((*grad1[0]), -0.4, 1e-3);
  EXPECT_NEAR((*grad1[1]), -0.03, 1e-3);
  EXPECT_NEAR((*grad1[2]), -0.1, 1e-3);
}

TEST(testgradient, forward_matrix_scale) {
  Graph g;
  std::mt19937 gen;
  std::uniform_real_distribution<> uniform{-10, 10};
  Eigen::MatrixXd m0 = Eigen::MatrixXd::Random(3, 2).array();
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(3, 2).array();
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(3, 2).array();
  auto m = g.add_constant_real_matrix(m0);
  g.get_node(m)->Grad1 = m1;
  g.get_node(m)->Grad2 = m2;

  double s0 = uniform(gen);
  double s1 = uniform(gen);
  double s2 = uniform(gen);
  auto s = g.add_constant_real(s0);
  g.get_node(s)->grad1 = s1;
  g.get_node(s)->grad2 = s2;

  auto scale = g.add_operator(OperatorType::MATRIX_SCALE, {s, m});
  g.get_node(scale)->eval(gen);
  g.get_node(scale)->compute_gradients();

  auto r0 = g.get_node(scale)->value._matrix;
  auto r1 = g.get_node(scale)->Grad1;
  auto r2 = g.get_node(scale)->Grad2;

  auto expected_r0 = s0 * m0;
  EXPECT_NEAR_MATRIX(expected_r0, r0);

  auto expected_r1 = s1 * m0 + s0 * m1;
  EXPECT_NEAR_MATRIX(expected_r1, r1);

  auto expected_r2 = s2 * m0 + 2 * s1 * m1 + s0 * m2;
  EXPECT_NEAR_MATRIX(expected_r2, r2);
}

TEST(testgradient, forward_broadcast_add) {
  Graph g;
  auto zero = g.add_constant_real(0.0);
  auto one = g.add_constant_pos_real(1.0);
  auto norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  auto norm_sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  auto nat_zero = g.add_constant_natural(0);
  auto nat_one = g.add_constant_natural(1);
  Eigen::MatrixXd m(2, 1);
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
  double grad1 = 0.0;
  double grad2 = 0.0;
  g.gradient_log_prob(norm_sample, grad1, grad2);
  EXPECT_NEAR(grad1, 1.0, 1e-3);
  EXPECT_NEAR(grad2, -4.5, 1e-3);
}

TEST(testgradient, backward_broadcast_add) {
  Graph g;
  auto zero = g.add_constant_real(0.0);
  auto one = g.add_constant_pos_real(1.0);
  auto two = g.add_constant_pos_real(2.0);
  auto five = g.add_constant_pos_real(5.0);

  auto stu_dist = g.add_distribution(
      DistributionType::STUDENT_T, AtomicType::REAL, {five, zero, two});
  auto stu_sample = g.add_operator(OperatorType::SAMPLE, {stu_dist});
  g.observe(stu_sample, 0.5);

  Eigen::MatrixXd m(2, 1);
  m << -1.0, 3.0;
  auto matrix = g.add_constant_real_matrix(m);
  auto sum_matrix =
      g.add_operator(OperatorType::BROADCAST_ADD, {stu_sample, matrix});
  auto nat_zero = g.add_constant_natural(0);
  auto nat_one = g.add_constant_natural(1);
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
  EXPECT_NEAR((*grad1[0]), -6.1481, 1e-3);
  EXPECT_NEAR((*grad1[1]), 3.0, 1e-3);
}

TEST(testgradient, matrix_add_grad) {
  Graph g;

  Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(3, 2);
  auto cm1 = g.add_constant_real_matrix(m1);
  Eigen::MatrixXd m1_grad1 = Eigen::MatrixXd::Random(3, 2);
  Eigen::MatrixXd m1_grad2 = Eigen::MatrixXd::Random(3, 2);
  g.get_node(cm1)->Grad1 = m1_grad1;
  g.get_node(cm1)->Grad2 = m1_grad2;

  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(3, 2);
  auto cm2 = g.add_constant_real_matrix(m2);
  Eigen::MatrixXd m2_grad1 = Eigen::MatrixXd::Random(3, 2);
  Eigen::MatrixXd m2_grad2 = Eigen::MatrixXd::Random(3, 2);
  g.get_node(cm2)->Grad1 = m2_grad1;
  g.get_node(cm2)->Grad2 = m2_grad2;

  auto result = g.add_operator(OperatorType::MATRIX_ADD, {cm1, cm2});
  auto rn = g.get_node(result);
  std::mt19937 gen;
  rn->eval(gen);
  rn->compute_gradients();

  auto grad1 = rn->Grad1;
  auto expected_grad1 = m1_grad1 + m2_grad1;
  EXPECT_NEAR_MATRIX(grad1, expected_grad1);

  auto grad2 = rn->Grad2;
  auto expected_grad2 = m1_grad2 + m2_grad2;
  EXPECT_NEAR_MATRIX(grad2, expected_grad2);

  // test where constant matrices don't have gradient initialized
  Graph g1;
  auto zero = g1.add_constant_real(0.0);
  auto one = g1.add_constant_pos_real(1.0);
  Eigen::MatrixXd constant(2, 1);
  constant << -2.0, 1.0;
  auto cm = g1.add_constant_real_matrix(constant);

  auto normal_dist = g1.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  auto two_natural = g1.add_constant_natural(2);
  auto sample = g1.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two_natural});
  auto add = g1.add_operator(OperatorType::MATRIX_ADD, {cm, sample});
  g1.query(add);

  auto cm_node = g1.get_node(cm);
  auto sample_node = g1.get_node(sample);
  sample_node->Grad1 = Eigen::MatrixXd::Ones(2, 1);
  sample_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
  auto add_node = g1.get_node(add);
  sample_node->eval(gen);
  sample_node->compute_gradients();
  add_node->eval(gen);
  add_node->compute_gradients();

  Eigen::MatrixXd expected_Grad1 = Eigen::MatrixXd::Ones(2, 1);
  Eigen::MatrixXd expected_Grad2 = Eigen::MatrixXd::Zero(2, 1);
  Eigen::MatrixXd Grad1 = add_node->Grad1;
  EXPECT_NEAR_MATRIX(Grad1, expected_Grad1);
  Eigen::MatrixXd Grad2 = add_node->Grad2;
  EXPECT_NEAR_MATRIX(Grad2, expected_Grad2);
}

TEST(testgradient, matrix_add_back_grad1) {
  Graph g;
  std::mt19937 gen;

  Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(3, 2);
  auto cm1 = g.add_constant_real_matrix(m1);
  auto r1 = g.add_operator(OperatorType::MATRIX_ADD, {cm1, cm1});
  Node* a = g.get_node(r1);
  a->eval(gen);

  auto result = g.add_operator(OperatorType::MATRIX_ADD, {r1, r1});
  Node* rn = g.get_node(result);
  rn->eval(gen);
  EXPECT_NEAR_MATRIX(rn->value._matrix, m1 * 4.0);

  a->reset_backgrad();
  rn->reset_backgrad();

  Eigen::MatrixXd grad = Eigen::MatrixXd::Random(3, 2);
  rn->back_grad1 += grad;
  EXPECT_NEAR_MATRIX(rn->back_grad1.as_matrix(), grad);

  rn->backward();
  EXPECT_NEAR_MATRIX(a->back_grad1.as_matrix(), grad * 2.0);
}

TEST(testgradient, matrix_negate_grad) {
  Graph g;

  Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(3, 2);
  auto cm1 = g.add_constant_real_matrix(m1);
  Eigen::MatrixXd m1_grad1 = Eigen::MatrixXd::Random(3, 2);
  Eigen::MatrixXd m1_grad2 = Eigen::MatrixXd::Random(3, 2);
  g.get_node(cm1)->Grad1 = m1_grad1;
  g.get_node(cm1)->Grad2 = m1_grad2;

  auto result = g.add_operator(OperatorType::MATRIX_NEGATE, {cm1});
  auto result_node = g.get_node(result);
  std::mt19937 gen;
  result_node->eval(gen);
  result_node->compute_gradients();

  auto grad1 = result_node->Grad1;
  auto expected_grad1 = -m1_grad1;
  EXPECT_NEAR_MATRIX(grad1, expected_grad1);

  auto grad2 = result_node->Grad2;
  auto expected_grad2 = -m1_grad2;
  EXPECT_NEAR_MATRIX(grad2, expected_grad2);
}

TEST(testgradient, matrix_negate_back_grad) {
  Graph g;
  std::mt19937 gen;

  Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(3, 2);
  auto cm = g.add_constant_real_matrix(m1);
  auto cm_node = g.get_node(cm);
  auto r1 = g.add_operator(OperatorType::MATRIX_ADD, {cm, cm});
  Node* rn1 = g.get_node(r1);
  rn1->eval(gen);

  auto result = g.add_operator(OperatorType::MATRIX_NEGATE, {r1});
  Node* result_node = g.get_node(result);
  result_node->eval(gen);
  EXPECT_NEAR_MATRIX(result_node->value._matrix, -2 * m1);

  rn1->reset_backgrad();
  result_node->reset_backgrad();

  Eigen::MatrixXd grad = Eigen::MatrixXd::Random(3, 2);
  result_node->back_grad1 += grad;
  EXPECT_NEAR_MATRIX(result_node->back_grad1.as_matrix(), grad);

  result_node->backward();
  EXPECT_NEAR_MATRIX(rn1->back_grad1.as_matrix(), -grad);
}

TEST(testgradient, backward_transpose) {
  /** Test gradients for a 3x3 matrix

     PyTorch validation code:

     x = tensor([[1.0, 0.98, 3.2], [0.2, 0.98, 1.0], [0.98, 0.2, 2.1]])
     transpose_sum = x.t().sum()
     log_p = (
         dist.Normal(transpose_sum, tensor(1.0)).log_prob(tensor(9.7))
         + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(x).sum()
     )
     autograd.grad(log_p, x)

   **/
  Graph g;
  auto zero = g.add_constant_real(0.0);
  auto pos1 = g.add_constant_pos_real(1.0);
  auto one = g.add_constant_natural(1);
  auto three = g.add_constant_natural(3);
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos1});

  auto sample = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, three});
  auto transpose =
      g.add_operator(OperatorType::TRANSPOSE, std::vector<uint>{sample});

  Eigen::MatrixXd x(3, 3);
  x << 1.0, 0.2, 0.98, 0.2, 0.98, 1.0, 0.98, 1.0, 2.1;

  g.observe(sample, x);

  // Uses two matrix multiplications to sum the result
  Eigen::MatrixXd m1(3, 1);
  m1 << 1.0, 1.0, 1.0;
  Eigen::MatrixXd m2(1, 3);
  m2 << 1.0, 1.0, 1.0;
  auto col_sum_m = g.add_constant_real_matrix(m1);
  auto row_sum_m = g.add_constant_real_matrix(m2);
  auto sum_rows = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{transpose, col_sum_m});
  auto sum_rows_and_cols = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{row_sum_m, sum_rows});
  auto sum_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{sum_rows_and_cols, pos1});

  auto sum_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{sum_dist});
  g.observe(sum_sample, 9.7);

  std::vector<DoubleMatrix*> grad(2);
  Eigen::MatrixXd expected_grad(3, 3);
  expected_grad << 0.26, 1.06, 0.28, 1.06, 0.28, 0.26, 0.28, 0.26, -0.84;

  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR_MATRIX(grad[0]->as_matrix(), expected_grad);
}

TEST(testgradient, forward_transpose) {
  Graph g;
  Eigen::MatrixXd x(3, 3), grad1(3, 3), grad2(3, 3);
  x << 1.0, 0.2, 0.98, 0.2, 0.98, 1.0, 0.98, 1.0, 2.1;
  grad1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
  grad2 << -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0;
  auto x_matrix = g.add_constant_real_matrix(x);
  Node* x_node = g.get_node(x_matrix);
  x_node->Grad1 = grad1;
  x_node->Grad2 = grad2;
  auto xt_matrix = g.add_operator(OperatorType::TRANSPOSE, {x_matrix});
  Node* xt = g.get_node(xt_matrix);
  std::mt19937 gen;
  xt->eval(gen);
  xt->compute_gradients();
  Eigen::MatrixXd first_grad = xt->Grad1;
  Eigen::MatrixXd second_grad = xt->Grad2;
  EXPECT_NEAR_MATRIX(first_grad, grad1.transpose());
  EXPECT_NEAR_MATRIX(second_grad, grad2.transpose());
}

TEST(testgradient, forward_cholesky) {
  // Test cholesky gradients for a 2x2 matrix

  // PyTorch verification
  // Sigma = torch.tensor([[1.0, 0.98], [0.98, 1.0]], requires_grad=True)
  // L = torch.cholesky(Sigma)
  // print(L)
  // a = grad(L[0, 0], Sigma, retain_graph=True, create_graph=True)[0].sum()
  // b = grad(L[0, 1], Sigma, retain_graph=True, create_graph=True)[0].sum()
  // c = grad(L[1, 0], Sigma, retain_graph=True, create_graph=True)[0].sum()
  // d = grad(L[1, 1], Sigma, retain_graph=True, create_graph=True)[0].sum()
  // a.requires_grad_(True)
  // b.requires_grad_(True)
  // c.requires_grad_(True)
  // d.requires_grad_(True)
  // aa = grad(a, Sigma, retain_graph=True)[0].sum()
  // bb = grad(b, Sigma, retain_graph=True)[0].sum()
  // cc = grad(c, Sigma, retain_graph=True)[0].sum()
  // dd = grad(d, Sigma, retain_graph=True)[0].sum()
  // first_grad <- [[a, b], [c, d]] = [[0.5, 0.0], [0.51, 0.001]]
  // second_grad <- [[aa, bb], [cc, dd]] = [[-0.25, 0.0], [-0.265, -0.002]]
  Graph g;
  Eigen::MatrixXd sigma(2, 2);
  sigma << 1.0, 0.98, 0.98, 1.0;
  auto sigma_matrix = g.add_constant_real_matrix(sigma);
  Node* sigma_node = g.get_node(sigma_matrix);
  sigma_node->Grad1 = Eigen::MatrixXd::Ones(2, 2);
  sigma_node->Grad2 = Eigen::MatrixXd::Zero(2, 2);
  auto l_matrix = g.add_operator(OperatorType::CHOLESKY, {sigma_matrix});
  Node* l_node = g.get_node(l_matrix);
  std::mt19937 gen;
  l_node->eval(gen);
  l_node->compute_gradients();
  Eigen::MatrixXd expected_first_grad(2, 2);
  expected_first_grad << 0.5, 0.0, 0.51, 0.001;
  Eigen::MatrixXd first_grad = l_node->Grad1;
  EXPECT_NEAR_MATRIX(first_grad, expected_first_grad);
  Eigen::MatrixXd expected_second_grad(2, 2);
  expected_second_grad << -0.25, 0.0, -0.265, -0.002;
  Eigen::MatrixXd second_grad = l_node->Grad2;
  EXPECT_NEAR_MATRIX(second_grad, expected_second_grad);

  // Sigma = torch.tensor([
  //     [1.0, 0.35, -0.25],
  //     [0.35, 1.54, 0.36],
  //     [-0.25, 0.36, 0.26]], requires_grad=True)
  // L = torch.cholesky(Sigma)
  // print(L)
  // grad1 = []
  // for i in range(3):
  //     row_vars = []
  //     for j in range(3):
  //         var = grad(L[i, j],
  //                    Sigma,
  //                    retain_graph=True,
  //                    create_graph=True)[0].sum()
  //         var.requires_grad_(True)
  //         row_vars.append(var)
  //     grad1.append(row_vars)
  // print(grad1)
  // grad2 = []
  // for i in range(3):
  //     row_vars = []
  //     for j in range(3):
  //         var = grad(grad1[i][j]
  //                    Sigma,
  //                    retain_graph=True,
  //                    create_graph=True)[0].sum()
  //         var.requires_grad_(True)
  //         row_vars.append(var)
  //     grad2.append(row_vars)
  // print(grad2)
  Graph g1;
  Eigen::MatrixXd sigma1(3, 3);
  sigma1 << 1.0, 0.35, -0.25, 0.35, 1.54, 0.36, -0.25, 0.36, 0.26;
  auto sigma1_matrix = g1.add_constant_real_matrix(sigma1);
  Node* sigma1_node = g1.get_node(sigma1_matrix);
  sigma1_node->Grad1 = Eigen::MatrixXd::Ones(3, 3);
  sigma1_node->Grad2 = Eigen::MatrixXd::Zero(3, 3);
  auto l1_matrix = g1.add_operator(OperatorType::CHOLESKY, {sigma1_matrix});
  Node* l1_node = g1.get_node(l1_matrix);
  l1_node->eval(gen);
  l1_node->compute_gradients();
  Eigen::MatrixXd expected_first_grad1(3, 3);
  expected_first_grad1 << 0.5, 0.0, 0.0, 0.825, 0.1774, 0.0, 1.125, 0.6264,
      2.3018;
  Eigen::MatrixXd first_grad1 = l1_node->Grad1;
  EXPECT_NEAR_MATRIX(first_grad1, expected_first_grad1);
  Eigen::MatrixXd expected_second_grad1(3, 3);
  expected_second_grad1 << -0.25, 0.0, 0.0, -0.7375, -0.3813, 0.0, -1.1875,
      -1.4312, -28.32;
  Eigen::MatrixXd second_grad1 = l1_node->Grad2;
  EXPECT_NEAR_MATRIX(second_grad1, expected_second_grad1);
}

TEST(testgradient, backward_cholesky) {
  /*

    PyTorch validation code:

    x = tensor([[1.0, 0.98, 3.2], [0.2, 0.98, 1.0], [0.98, 0.2, 2.1]],
            requires_grad=True)
    choleskySum = cholesky(x).sum()
    log_p = (
        dist.Normal(choleskySum, tensor(1.0)).log_prob(tensor(1.7))
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(x).sum()
    )
    autograd.grad(log_p, x)
  */
  Graph g;
  auto zero = g.add_constant_real(0.0);
  auto pos1 = g.add_constant_pos_real(1.0);
  auto one = g.add_constant_natural(1);
  auto three = g.add_constant_natural(3);
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos1});

  auto sigma_sample = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, three});
  auto L =
      g.add_operator(OperatorType::CHOLESKY, std::vector<uint>{sigma_sample});

  Eigen::MatrixXd sigma(3, 3);
  sigma << 1.0, 0.2, 0.98, 0.2, 0.98, 1.0, 0.98, 1.0, 2.1;

  g.observe(sigma_sample, sigma);

  // Uses two matrix multiplications to sum the result of Cholesky
  auto col_sum_m = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, one});
  auto row_sum_m = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, one, three});
  Eigen::MatrixXd m1(3, 1);
  m1 << 1.0, 1.0, 1.0;
  Eigen::MatrixXd m2(1, 3);
  m2 << 1.0, 1.0, 1.0;
  g.observe(col_sum_m, m1);
  g.observe(row_sum_m, m2);
  auto sum_rows = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{L, col_sum_m});
  auto sum_chol = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{row_sum_m, sum_rows});
  auto sum_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{sum_chol, pos1});

  auto sum_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{sum_dist});
  g.observe(sum_sample, 1.7);

  std::vector<DoubleMatrix*> grad(4);
  Eigen::MatrixXd expected_grad(3, 3);
  expected_grad << -2.7761, -1.6587, -0.3756, -1.6587, -2.8059, -0.6445,
      -0.3756, -0.6445, -4.2949;

  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 4);
  EXPECT_NEAR_MATRIX(grad[0]->as_matrix(), expected_grad);
}

TEST(testgradient, matrix_exp_grad) {
  Graph g;

  Eigen::MatrixXd m1(1, 2);
  m1 << -2., -3.0;
  auto cm1 = g.add_constant_real_matrix(m1);

  auto c = g.add_constant_real(2.0);
  auto cm = g.add_operator(OperatorType::MATRIX_SCALE, {c, cm1});
  auto exp = g.add_operator(OperatorType::MATRIX_EXP, {cm});

  // test forward
  // TODO: hide tests inside graph helper function
  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(1, 2);
  cm1_node->Grad2 = Eigen::MatrixXd::Zero(1, 2);
  Node* cm_node = g.get_node(cm);
  Node* exp_node = g.get_node(exp);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();
  exp_node->eval(gen);
  exp_node->compute_gradients();
  Eigen::MatrixXd first_grad = exp_node->Grad1;
  Eigen::MatrixXd expected_first_grad(1, 2);
  expected_first_grad << 0.0366313, 0.0049575;
  EXPECT_NEAR_MATRIX(first_grad, expected_first_grad);
  Eigen::MatrixXd second_grad = exp_node->Grad2;
  Eigen::MatrixXd expected_second_grad(1, 2);
  expected_second_grad << 0.0732626, 0.00991501;
  EXPECT_NEAR_MATRIX(second_grad, expected_second_grad);

  // test backward
  Graph g1;
  auto zero = g1.add_constant_real(0.0);
  auto one = g1.add_constant_pos_real(1.0);
  auto x_dist = g1.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  auto two = g1.add_constant_natural(2);
  auto x_sample =
      g1.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{x_dist, two});
  Eigen::MatrixXd x_value(2, 1);
  x_value << 1.0, 0.5;
  g1.observe(x_sample, x_value);
  auto exp_x_pos = g1.add_operator(OperatorType::MATRIX_EXP, {x_sample});
  auto exp_x = g1.add_operator(OperatorType::TO_REAL_MATRIX, {exp_x_pos});

  auto index_zero = g1.add_constant_natural(0);
  auto exp_x1 = g1.add_operator(OperatorType::INDEX, {exp_x, index_zero});
  auto y1_dist = g1.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {exp_x1, one});
  auto y1_sample =
      g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{y1_dist});
  g1.observe(y1_sample, 2.5);

  auto index_one = g1.add_constant_natural(1);
  auto exp_x2 = g1.add_operator(OperatorType::INDEX, {exp_x, index_one});
  auto y2_dist = g1.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {exp_x2, one});
  auto y2_sample =
      g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{y2_dist});
  g1.observe(y2_sample, 1.5);

  /*
  PyTorch verification
  x_dist = dist.Normal(0, 1)
  x1_sample = tensor(1.0, requires_grad=True)
  exp_x1 = torch.exp(x1_sample)
  y1_dist = dist.Normal(exp_x1, 1.0)
  y1_sample = tensor(2.5)
  x2_sample = tensor(0.5, requires_grad=True)
  exp_x2 = torch.exp(x2_sample)
  y2_dist = dist.Normal(exp_x2, 1.0)
  y2_sample = tensor(1.5)
  log_prob = y1_dist.log_prob(y1_sample) + y2_dist.log_prob(y2_sample)
            + x_dist.log_prob(x1_sample) + x_dist.log_prob(x2_sample)

  grad(log_prob, x1_sample) -> -1.5934
  grad(log_prob, x2_sample) -> -0.7452
  grad(log_prob, y1_sample) -> 0.2183
  grad(log_prob, y2_sample) -> 0.1487
  */

  std::vector<DoubleMatrix*> grad1;
  g1.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), -1.5934, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), -0.7452, 1e-3);
  EXPECT_NEAR((*grad1[1]), 0.2183, 1e-3);
  EXPECT_NEAR((*grad1[2]), 0.1487, 1e-3);
}

TEST(testgradient, matrix_log_grad_forward) {
  Graph g;

  // Test forward differentiation

  Eigen::MatrixXd m1(1, 2);
  m1 << 2.0, 3.0;
  auto cm1 = g.add_constant_pos_matrix(m1);
  auto c = g.add_constant_pos_real(2.0);
  auto cm = g.add_operator(OperatorType::MATRIX_SCALE, {c, cm1});
  auto mlog = g.add_operator(OperatorType::MATRIX_LOG, {cm});
  // f(x) = log(2 * g(x))
  // g(x) = x, x = [2, 3]
  // hence we set
  // g'(x) = 1
  // g''(x) = 0
  // for testing.

  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(1, 2);
  cm1_node->Grad2 = Eigen::MatrixXd::Zero(1, 2);
  Node* cm_node = g.get_node(cm);
  Node* mlog_node = g.get_node(mlog);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();
  mlog_node->eval(gen);
  mlog_node->compute_gradients();
  Eigen::MatrixXd first_grad = mlog_node->Grad1;
  Eigen::MatrixXd expected_first_grad(1, 2);
  // By chain rule, f'(x) should be 2 * g'(x) / 2 * g(x) = [0.5, 0.33]
  expected_first_grad << 0.5, 1.0 / 3.0;
  EXPECT_NEAR_MATRIX(first_grad, expected_first_grad);
  Eigen::MatrixXd second_grad = mlog_node->Grad2;
  Eigen::MatrixXd expected_second_grad(1, 2);
  // f''(x) = (g''(x) * g'(x) + g'(x) * g'(x)) / (g(x) * g(x))
  // = ([0, 0] * [1, 1] - [1, 1] * [1, 1]) / ([2, 3] * [2, 3])
  // = [-0.25, -0.11]
  expected_second_grad << -0.25, -1.0 / 9.0;
  EXPECT_NEAR_MATRIX(second_grad, expected_second_grad);
}

TEST(testgradient, matrix_log_grad_backward) {
  /*
# Test backward differentiation
#
# Build the same model in PyTorch and BMG; we should get the same
# backwards gradients as PyTorch.

import torch
hn = torch.distributions.HalfNormal(1)
s0 = torch.tensor(1.0, requires_grad=True)
s1 = torch.tensor(0.5, requires_grad=True)
mlog0 = s0.log()
mlog1 = s1.log()
n0 = torch.distributions.Normal(mlog0, 1.0)
n1 = torch.distributions.Normal(mlog1, 1.0)
sn0 = torch.tensor(2.5, requires_grad=True)
sn1 = torch.tensor(1.5, requires_grad=True)
log_prob = (n0.log_prob(sn0) + n1.log_prob(sn1) +
  hn.log_prob(s0) + hn.log_prob(s1))
torch.autograd.grad(log_prob, s0, retain_graph=True) # 1.5000
torch.autograd.grad(log_prob, s1, retain_graph=True) # 3.8863
torch.autograd.grad(log_prob, sn0, retain_graph=True) # -2.5000
torch.autograd.grad(log_prob, sn1, retain_graph=True) # -2.1931
  */

  Graph g;
  auto one = g.add_constant_pos_real(1.0);
  auto hn = g.add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::POS_REAL, {one});
  auto two = g.add_constant_natural(2);
  auto hn_sample =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{hn, two});
  Eigen::MatrixXd hn_observed(2, 1);
  hn_observed << 1.0, 0.5;
  g.observe(hn_sample, hn_observed);

  auto mlog_pos = g.add_operator(OperatorType::MATRIX_LOG, {hn_sample});
  auto mlog = g.add_operator(OperatorType::TO_REAL_MATRIX, {mlog_pos});
  auto index_zero = g.add_constant_natural(0);
  auto mlog0 = g.add_operator(OperatorType::INDEX, {mlog, index_zero});
  auto index_one = g.add_constant_natural(1);
  auto mlog1 = g.add_operator(OperatorType::INDEX, {mlog, index_one});

  auto n0 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog0, one});
  auto n1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog1, one});

  auto ns0 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n0});
  g.observe(ns0, 2.5);
  auto ns1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n1});
  g.observe(ns1, 1.5);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), 1.5000, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), 3.8863, 1e-3);
  EXPECT_NEAR((*grad1[1]), -2.5000, 1e-3);
  EXPECT_NEAR((*grad1[2]), -2.1931, 1e-3);
}

TEST(testgradient, matrix_phi_grad_forward) {
  Graph g;

  // Test forward differentiation

  Eigen::MatrixXd m1(1, 2);
  m1 << 2.0, 3.0;
  auto cm1 = g.add_constant_real_matrix(m1);
  auto c = g.add_constant_real(2.0);
  auto cm = g.add_operator(OperatorType::MATRIX_SCALE, {c, cm1});
  auto mphi = g.add_operator(OperatorType::MATRIX_PHI, {cm});
  // f(x) = phi(2 * g(x))
  // g(x) = [2, 3]
  // but we artificially set
  // g'(x) = [1, 1]
  // g''(x) = [0, 0]
  // for testing.

  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(1, 2);
  cm1_node->Grad2 = Eigen::MatrixXd::Zero(1, 2);
  Node* cm_node = g.get_node(cm);
  Node* mphi_node = g.get_node(mphi);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();
  mphi_node->eval(gen);
  mphi_node->compute_gradients();

  // First derivative:
  Eigen::MatrixXd first_grad = mphi_node->Grad1;
  Eigen::MatrixXd h(1, 2);
  Eigen::MatrixXd expected_first_grad(1, 2);
  // h(x) = 2 * exp(-2 g(x)^2) / sqrt(2pi)
  // f'(x) = g'(x) * h(x)
  // But g'(x) = [1, 1], so f'(x) == h(x)
  double h0 = beanmachine::oper::_1_SQRT2PI * 2.0 * std::exp(-8.0);
  double h1 = beanmachine::oper::_1_SQRT2PI * 2.0 * std::exp(-18.0);
  expected_first_grad << h0, h1;
  EXPECT_NEAR_MATRIX(first_grad, expected_first_grad);

  // Second derivative:
  // Turns out that in this case where g' is [1, 1]
  // and g'' is [0, 0]:
  // f''(x) = - 4 g(x) f'(x)
  Eigen::MatrixXd second_grad = mphi_node->Grad2;
  Eigen::MatrixXd expected_second_grad(1, 2);
  expected_second_grad << -h0 * 8.0, -h1 * 12.0;
  EXPECT_NEAR_MATRIX(second_grad, expected_second_grad);
}

TEST(testgradient, matrix_elementwise_mult_forward) {
  Graph g;

  Eigen::MatrixXd m1(3, 2);
  m1 << 0.3, -0.1, 1.2, 0.9, -2.6, 0.8;
  auto cm1 = g.add_constant_real_matrix(m1);

  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(3, 2);
  cm1_node->Grad2 = Eigen::MatrixXd::Ones(3, 2);

  auto cm = g.add_operator(OperatorType::ELEMENTWISE_MULTIPLY, {cm1, cm1});
  Node* cm_node = g.get_node(cm);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();

  Eigen::MatrixXd first_grad_x = cm_node->Grad1;
  Eigen::MatrixXd expected_first_grad_x(3, 2);
  expected_first_grad_x << 0.6, -0.2, 2.4, 1.8, -5.2, 1.6;
  EXPECT_NEAR_MATRIX(first_grad_x, expected_first_grad_x);
  Eigen::MatrixXd second_grad_x = cm_node->Grad2;
  Eigen::MatrixXd expected_second_grad_x(3, 2);
  expected_second_grad_x << 2.6, 1.8, 4.4, 3.8, -3.2, 3.6;
  EXPECT_NEAR_MATRIX(second_grad_x, expected_second_grad_x);
}

TEST(testgradient, matrix_phi_grad_backward) {
  /*
# Test backward differentiation
#
# Build the same model in PyTorch and BMG; we should get the same
# backwards gradients as PyTorch.

import torch
phi = torch.distributions.Normal(0.0, 1.0).cdf
hn = torch.distributions.HalfNormal(1)
s0 = torch.tensor(1.0, requires_grad=True)
s1 = torch.tensor(0.5, requires_grad=True)
mphi0 = phi(s0)
mphi1 = phi(s1)
n0 = torch.distributions.Normal(mphi0, 1.0)
n1 = torch.distributions.Normal(mphi1, 1.0)
sn0 = torch.tensor(2.5, requires_grad=True)
sn1 = torch.tensor(1.5, requires_grad=True)
log_prob = (n0.log_prob(sn0) + n1.log_prob(sn1) +
  hn.log_prob(s0) + hn.log_prob(s1))
torch.autograd.grad(log_prob, s0, retain_graph=True) # -0.5987
torch.autograd.grad(log_prob, s1, retain_graph=True) # -0.2153
torch.autograd.grad(log_prob, sn0, retain_graph=True) # -1.6587
torch.autograd.grad(log_prob, sn1, retain_graph=True) # -0.8085
  */

  Graph g;
  auto one = g.add_constant_pos_real(1.0);
  auto hn = g.add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::POS_REAL, {one});
  auto two = g.add_constant_natural(2);
  auto hn_sample =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{hn, two});
  Eigen::MatrixXd hn_observed(2, 1);
  hn_observed << 1.0, 0.5;
  g.observe(hn_sample, hn_observed);

  auto to_real = g.add_operator(OperatorType::TO_REAL_MATRIX, {hn_sample});
  auto mlog_pos = g.add_operator(OperatorType::MATRIX_PHI, {to_real});
  auto mlog = g.add_operator(OperatorType::TO_REAL_MATRIX, {mlog_pos});
  auto index_zero = g.add_constant_natural(0);
  auto mlog0 = g.add_operator(OperatorType::INDEX, {mlog, index_zero});
  auto index_one = g.add_constant_natural(1);
  auto mlog1 = g.add_operator(OperatorType::INDEX, {mlog, index_one});

  auto n0 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog0, one});
  auto n1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog1, one});

  auto ns0 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n0});
  g.observe(ns0, 2.5);
  auto ns1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n1});
  g.observe(ns1, 1.5);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), -0.5987, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), -0.2153, 1e-3);
  EXPECT_NEAR((*grad1[1]), -1.6587, 1e-3);
  EXPECT_NEAR((*grad1[2]), -0.8085, 1e-3);
}

TEST(testgradient, matrix_elementwise_mult_backward) {
  Graph g;
  auto zero = g.add_constant_real(0.0);
  auto pos1 = g.add_constant_pos_real(1.0);
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos1});
  auto one = g.add_constant_natural(1);
  auto two = g.add_constant_natural(2);
  auto three = g.add_constant_natural(3);

  Eigen::MatrixXd m1(3, 2);
  Eigen::MatrixXd m2(3, 2);
  m1 << 0.3, -0.1, 1.2, 0.9, -2.6, 0.8;
  m2 << 0.4, 0.1, 0.5, -1.1, 0.7, -0.6;

  auto x = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, two});
  auto y = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, two});
  auto xy = g.add_operator(
      OperatorType::ELEMENTWISE_MULTIPLY, std::vector<uint>{x, y});

  g.observe(x, m1);
  g.observe(y, m2);

  // test backwards
  //
  // Pytorch validation:
  // X = tensor([[0.3, -0.1], [1.2, 0.9], [-2.6, 0.8]], requires_grad=True)
  // Y = tensor([[0.4, 0.1], [0.5, -1.1], [0.7, -0.6]], requires_grad=True)
  // def f_grad(x):
  //   XYSum = torch.mul(X, Y).sum()
  //   log_p = (
  //       dist.Normal(XYSum, tensor(1.0)).log_prob(tensor(1.7))
  //       dist.Normal(tensor(0.0), tensor(1.0)).log_prob(X).sum()
  //       dist.Normal(tensor(0.0), tensor(1.0)).log_prob(Y).sum()
  //   )
  //  return torch.autograd.grad(log_p, x)[0]
  //
  // result:
  // [tensor([[ 1.4120,  0.5280],
  //          [ 0.9400, -5.6080],
  //       [ 5.5960, -3.3680]]),
  // tensor([[  0.8840,  -0.5280],
  //        [  4.6360,   4.9520],
  //        [-11.8280,   4.0240]])]

  // Uses two matrix multiplications to sum the result of xy
  auto col_sum_m = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two, one});
  auto row_sum_m = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, one, three});
  Eigen::MatrixXd m5(2, 1);
  m5 << 1.0, 1.0;
  Eigen::MatrixXd m6(1, 3);
  m6 << 1.0, 1.0, 1.0;
  g.observe(col_sum_m, m5);
  g.observe(row_sum_m, m6);
  auto xy_sum_rows = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{xy, col_sum_m});
  auto xy_sum = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{row_sum_m, xy_sum_rows});
  auto xy_sum_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{xy_sum, pos1});

  auto xyz_sum_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{xy_sum_dist});
  g.observe(xyz_sum_sample, 1.7);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 5);
  // grad x
  EXPECT_NEAR(grad1[0]->coeff(0), 1.4120, 1e-3);
  EXPECT_NEAR(grad1[0]->coeff(1), 0.9400, 1e-3);
  EXPECT_NEAR(grad1[0]->coeff(2), 5.5960, 1e-3);
  EXPECT_NEAR(grad1[0]->coeff(3), 0.5280, 1e-3);
  EXPECT_NEAR(grad1[0]->coeff(4), -5.6080, 1e-3);
  EXPECT_NEAR(grad1[0]->coeff(5), -3.3680, 1e-3);
  // grad y
  EXPECT_NEAR(grad1[1]->coeff(0), 0.8840, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(1), 4.6360, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(2), -11.8280, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(3), -0.5280, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(4), 4.9520, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(5), 4.0240, 1e-3);
}

TEST(testgradient, matrix_mult_forward) {
  Graph g;

  Eigen::MatrixXd m1(2, 3);
  m1 << 0.3, -0.1, 1.2, 0.9, -2.6, 0.8;
  auto cm1 = g.add_constant_real_matrix(m1);

  Eigen::MatrixXd m2(3, 2);
  m2 << 0.8, 1.1, 0.4, 4.3, 1.0, 0.8;
  auto cm2 = g.add_constant_real_matrix(m2);

  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(2, 3);
  cm1_node->Grad2 = Eigen::MatrixXd::Ones(2, 3);

  Node* cm2_node = g.get_node(cm2);
  cm2_node->Grad1 = Eigen::MatrixXd::Ones(3, 2);
  cm2_node->Grad2 = 0.5 * Eigen::MatrixXd::Ones(3, 2);

  auto cm = g.add_operator(OperatorType::MATRIX_MULTIPLY, {cm1, cm2});
  Node* cm_node = g.get_node(cm);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();

  Eigen::MatrixXd first_grad_x = cm_node->Grad1;
  Eigen::MatrixXd expected_first_grad_x(2, 2);
  expected_first_grad_x << 3.6, 7.6, 1.3, 5.3;
  EXPECT_NEAR_MATRIX(first_grad_x, expected_first_grad_x);
  Eigen::MatrixXd second_grad_x = cm_node->Grad2;
  Eigen::MatrixXd expected_second_grad_x(2, 2);
  expected_second_grad_x << 8.9, 12.9, 7.75, 11.75;
  EXPECT_NEAR_MATRIX(second_grad_x, expected_second_grad_x);

  // node does not have Grad initialized
  Graph g1;
  Eigen::MatrixXd mat1(2, 2);
  mat1 << 0.5, 2.0, -1.0, -0.2;
  auto cmat1 = g1.add_constant_real_matrix(mat1);
  Eigen::MatrixXd mat2(2, 2);
  mat2 << 0.6, 2.5, 1.0, 0.4;
  auto cmat2 = g1.add_constant_real_matrix(mat2);
  auto mult = g1.add_operator(OperatorType::MATRIX_MULTIPLY, {cmat1, cmat2});

  auto mat2_node = g1.get_node(cmat2);
  mat2_node->Grad1 = Eigen::MatrixXd::Ones(2, 2);
  mat2_node->Grad2 = Eigen::MatrixXd::Ones(2, 2) * 3;
  mat2_node->compute_gradients();
  auto mult_node = g1.get_node(mult);
  mult_node->eval(gen);
  mult_node->compute_gradients();
  Eigen::MatrixXd first_grad = mult_node->Grad1;
  Eigen::MatrixXd expected_first_grad(2, 2);
  expected_first_grad << 2.5, 2.5, -1.2, -1.2;
  EXPECT_NEAR_MATRIX(first_grad, expected_first_grad);
  Eigen::MatrixXd second_grad = mult_node->Grad2;
  Eigen::MatrixXd expected_second_grad(2, 2);
  expected_second_grad << 7.5, 7.5, -3.6, -3.6;
  EXPECT_NEAR_MATRIX(second_grad, expected_second_grad);
}

TEST(testgradient, log_prob) {
  std::mt19937 gen;
  Graph g;
  double epsilon = 0.0000001;
  auto zero = g.add_constant_real(0.0);
  auto pos_zero = g.add_constant_pos_real(0.0);

  auto makeConst = [&](double value) {
    auto v = g.add_constant_real(value);
    auto plus = g.add_operator(OperatorType::ADD, {zero, v});
    g.get_node(plus)->eval(gen);
    return plus;
  };

  auto makePosConst = [&](double value) {
    auto v = g.add_constant_pos_real(value);
    auto plus = g.add_operator(OperatorType::ADD, {pos_zero, v});
    g.get_node(plus)->eval(gen);
    return plus;
  };

  auto oneTest = [&](double value, double mean, double stdev) {
    auto value_node = g.get_node(makeConst(value));
    auto mean_node = g.get_node(makeConst(mean));
    auto stdev_node = g.get_node(makePosConst(stdev));
    auto distribution = g.add_distribution(
        DistributionType::NORMAL,
        AtomicType::REAL,
        {mean_node->index, stdev_node->index});
    auto logprob_node = g.get_node(g.add_operator(
        OperatorType::LOG_PROB, {distribution, value_node->index}));
    auto clear_gradient = [&](Node* node) {
      node->grad1 = 0;
      node->grad2 = 0;
      node->back_grad1 = 0;
    };
    auto clear_gradients = [&]() {
      clear_gradient(value_node);
      clear_gradient(mean_node);
      clear_gradient(stdev_node);
      clear_gradient(logprob_node);
    };

    // test gradient1 with respect to mean
    clear_gradients();
    mean_node->grad1 = 1;
    logprob_node->compute_gradients();
    // See https://www.wolframalpha.com/
    // D[Log[PDF[NormalDistribution[m, s], v]], m] => (-m + v)/s^2
    EXPECT_NEAR(
        logprob_node->grad1, (-mean + value) / (stdev * stdev), epsilon);
    auto gradient_vs_mean1 = logprob_node->grad1;

    // test gradient2 with respect to mean
    // D[D[log(PDF(NormalDistribution[m, s], v)), m], m] => -s^(-2)
    EXPECT_NEAR(logprob_node->grad2, -pow(stdev, -2), epsilon);

    // test gradient1 with respect to stdev
    clear_gradients();
    stdev_node->grad1 = 1;
    logprob_node->compute_gradients();
    // D[Log[PDF[NormalDistribution[m, s], v]], s] =>
    //      (m^2 - s^2 - 2 m v + v^2)/s^3
    EXPECT_NEAR(
        logprob_node->grad1,
        (pow(mean, 2) - pow(stdev, 2) - 2 * mean * value + pow(value, 2)) /
            pow(stdev, 3),
        epsilon);
    auto gradient_vs_stdev1 = logprob_node->grad1;

    // test gradient2 with respect to stdev
    // D[D[log(PDF(NormalDistribution[m, s], v)), s], s] =>
    //        (-3 m^2 + s^2 + 6 m v - 3 v^2)/s^4
    EXPECT_NEAR(
        logprob_node->grad2,
        (-3 * pow(mean, 2) + pow(stdev, 2) + 6 * mean * value -
         3 * pow(value, 2)) /
            pow(stdev, 4),
        epsilon);

    // test gradient1 with respect to value
    clear_gradients();
    value_node->grad1 = 1;
    logprob_node->compute_gradients();
    // D[log(PDF(NormalDistribution[m, s], v)), v] => (m - v)/s^2
    EXPECT_NEAR(logprob_node->grad1, (mean - value) / (stdev * stdev), epsilon);
    auto gradient_vs_value1 = logprob_node->grad1;

    // test gradient2 with respect to value
    // D[D[Log[PDF[NormalDistribution[m, s], v]], v], v] => -s^(-2)
    EXPECT_NEAR(logprob_node->grad2, -pow(stdev, -2), epsilon);
    auto gradient_vs_value2 = logprob_node->grad2;

    // test for proper application of the chain rule (vs value).
    clear_gradients();
    value_node->grad1 = 1.1;
    value_node->grad2 = 2.2;
    logprob_node->compute_gradients();
    EXPECT_NEAR(
        logprob_node->grad1, value_node->grad1 * gradient_vs_value1, epsilon);
    EXPECT_NEAR(
        logprob_node->grad2,
        gradient_vs_value2 * pow(value_node->grad1, 2) +
            value_node->grad2 * gradient_vs_value1,
        epsilon);

    // test for backward gradients.
    clear_gradients();
    logprob_node->back_grad1 = 1.1;
    logprob_node->backward();
    EXPECT_NEAR(
        value_node->back_grad1,
        gradient_vs_value1 * logprob_node->back_grad1,
        epsilon);
    EXPECT_NEAR(
        stdev_node->back_grad1,
        gradient_vs_stdev1 * logprob_node->back_grad1,
        epsilon);
    EXPECT_NEAR(
        mean_node->back_grad1,
        gradient_vs_mean1 * logprob_node->back_grad1,
        epsilon);
  };

  auto values = std::vector<double>({-1.2, 0, 2.13, 5});
  for (const double value : values) {
    for (const double mean : values) {
      for (const double stdev : values) {
        if (stdev != 0) {
          oneTest(value, mean, std::abs(stdev));
        }
      }
    }
  }
}

TEST(testgradient, matrix_sum) {
  Graph g;
  std::mt19937 gen;

  // forward mode
  // x = [[2.0, 4.0], [1.0, -3.0]]
  // square = x ^ 2 (element-wise)
  // sum = x.sum()
  Eigen::MatrixXd m1(2, 2);
  m1 << 2.0, 4.0, 1.0, -3.0;
  auto cm1 = g.add_constant_real_matrix(m1);
  auto cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(2, 2);
  cm1_node->Grad2 = Eigen::MatrixXd::Zero(2, 2);
  auto square = g.add_operator(OperatorType::ELEMENTWISE_MULTIPLY, {cm1, cm1});
  auto sum = g.add_operator(OperatorType::MATRIX_SUM, {square});

  Node* square_node = g.get_node(square);
  square_node->eval(gen);
  square_node->compute_gradients();
  Node* sum_node = g.get_node(sum);
  sum_node->eval(gen);
  sum_node->compute_gradients();

  double first_grad = sum_node->grad1;
  double second_grad = sum_node->grad2;
  EXPECT_NEAR(first_grad, 8.0, 1e-5);
  EXPECT_NEAR(second_grad, 8.0, 1e-5);

  // backward mode
  // x ~ Normal(0, 1)
  // y ~ Normal([x, x^2].sum(), 1)
  // PyTorch verification
  // x = tensor(0.5, requires_grad=True)
  // x_sum = x + x.pow(2) # equivalent to [x, x**2].sum()
  // y = tensor(2.0, requires_grad=True)
  // log_p = dist.Normal(0, 1).log_prob(x) + dist.Normal(x_sum, 1).log_prob(y)
  // grad(log_p, x) -> 2.0
  // grad(log_p, y) -> -1.25

  Graph g1;
  auto zero = g1.add_constant_real(0.0);
  auto pos_one = g1.add_constant_pos_real(1.0);
  auto x_dist = g1.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos_one});
  auto x = g1.add_operator(OperatorType::SAMPLE, {x_dist});
  g1.observe(x, 0.5);
  auto x_squared = g1.add_operator(OperatorType::MULTIPLY, {x, x});
  auto one = g1.add_constant_natural(1);
  auto two = g1.add_constant_natural(2);
  auto x_matrix =
      g1.add_operator(OperatorType::TO_MATRIX, {two, one, x, x_squared});
  auto x_sum = g1.add_operator(OperatorType::MATRIX_SUM, {x_matrix});
  auto y_dist = g1.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{x_sum, pos_one});
  auto y = g1.add_operator(OperatorType::SAMPLE, {y_dist});
  g1.observe(y, 2.0);

  std::vector<DoubleMatrix*> grad1;
  g1.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR((*grad1[0]), 2.0, 1e-3);
  EXPECT_NEAR((*grad1[1]), -1.25, 1e-3);
}

TEST(testgradient, matrix_log1p_grad_forward) {
  Graph g;

  // Test forward differentiation

  Eigen::MatrixXd g0(1, 2);
  g0 << 2.0, 3.0;
  auto cm1 = g.add_constant_pos_matrix(g0);
  auto c = g.add_constant_pos_real(2.0);
  auto cm = g.add_operator(OperatorType::MATRIX_SCALE, {c, cm1});
  auto mlog1p = g.add_operator(OperatorType::MATRIX_LOG1P, {cm});
  // f(x) = log1p(2 * g(x))
  // g(x) = [2, 3]
  // but we artificially set
  // g'(x) = [1.1, 1.1]
  // g''(x) = [2.3, 2.3]
  // for testing.

  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(1, 2) * 1.1;
  auto g1 = cm1_node->Grad1.array();
  cm1_node->Grad2 = Eigen::MatrixXd::Ones(1, 2) * 2.3;
  auto g2 = cm1_node->Grad2.array();
  Node* cm_node = g.get_node(cm);
  Node* mlog1p_node = g.get_node(mlog1p);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();
  mlog1p_node->eval(gen);
  mlog1p_node->compute_gradients();

  // For debugging, check the gradients of cm_node
  EXPECT_NEAR_MATRIX(g1 * 2, cm_node->Grad1.array());
  EXPECT_NEAR_MATRIX(g2 * 2, cm_node->Grad2.array());

  // First derivative:
  auto first_grad = mlog1p_node->Grad1.array();
  // f = log1p(2 * g) = log(1 + 2 * g)
  // f' = (2 g') / (1 + 2 * g)
  auto f1top = 2 * g1;
  auto f1bot = 1 + 2 * g0.array();
  auto expected_f1 = f1top / f1bot;
  EXPECT_NEAR_MATRIX(expected_f1, first_grad);

  // Second derivative:
  auto second_grad = mlog1p_node->Grad2.array();
  // f' = (2 g') / (1 + 2 g) = f1top / f1bot
  // where f1top = 2 g'
  //       f1bot = 1 + 2 g
  // f'' = (f1bot f1top' - f1top f1bot') / f1bot^2
  // where f1top' = 2 g''
  //       f1bot' = 2 g' = f1top
  auto f1top1 = 2 * g2;
  auto f1bot1 = f1top;
  auto expected_f2 = (f1bot * f1top1 - f1top * f1bot1) / f1bot.pow(2);
  EXPECT_NEAR_MATRIX(expected_f2, second_grad);
}

TEST(testgradient, matrix_log1p_grad_backward) {
  /*
# Test backward differentiation
#
# Build the same model in PyTorch and BMG; we should get the same
# backwards gradients as PyTorch.

import torch
hn = torch.distributions.HalfNormal(1)
s0 = torch.tensor(1.0, requires_grad=True)
s1 = torch.tensor(0.5, requires_grad=True)
mlog0 = s0.log1p()
mlog1 = s1.log1p()
n0 = torch.distributions.Normal(mlog0, 1.0)
n1 = torch.distributions.Normal(mlog1, 1.0)
sn0 = torch.tensor(2.5, requires_grad=True)
sn1 = torch.tensor(1.5, requires_grad=True)
log_prob = (n0.log_prob(sn0) + n1.log_prob(sn1) +
  hn.log_prob(s0) + hn.log_prob(s1))
print(torch.autograd.grad(log_prob, s0, retain_graph=True)) # -0.0966
print(torch.autograd.grad(log_prob, s1, retain_graph=True)) # 0.2297
print(torch.autograd.grad(log_prob, sn0, retain_graph=True)) # -1.8069
print(torch.autograd.grad(log_prob, sn1, retain_graph=True)) # -1.0945
  */

  Graph g;
  auto one = g.add_constant_pos_real(1.0);
  auto hn = g.add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::POS_REAL, {one});
  auto two = g.add_constant_natural(2);
  auto hn_sample =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{hn, two});
  Eigen::MatrixXd hn_observed(2, 1);
  hn_observed << 1.0, 0.5;
  g.observe(hn_sample, hn_observed);

  auto mlog_pos = g.add_operator(OperatorType::MATRIX_LOG1P, {hn_sample});
  auto mlog = g.add_operator(OperatorType::TO_REAL_MATRIX, {mlog_pos});
  auto index_zero = g.add_constant_natural(0);
  auto mlog0 = g.add_operator(OperatorType::INDEX, {mlog, index_zero});
  auto index_one = g.add_constant_natural(1);
  auto mlog1 = g.add_operator(OperatorType::INDEX, {mlog, index_one});

  auto n0 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog0, one});
  auto n1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog1, one});

  auto ns0 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n0});
  g.observe(ns0, 2.5);
  auto ns1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n1});
  g.observe(ns1, 1.5);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), -0.0966, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), 0.2297, 1e-3);
  EXPECT_NEAR((*grad1[1]), -1.8069, 1e-3);
  EXPECT_NEAR((*grad1[2]), -1.0945, 1e-3);
}

TEST(testgradient, matrix_log1mexp_grad_forward) {
  Graph g;

  // Test forward differentiation

  Eigen::MatrixXd _g0(1, 2);
  _g0 << 2.0, 3.0;
  auto cm1 = g.add_constant_pos_matrix(_g0);
  auto g0 = _g0.array();
  auto c = g.add_constant_pos_real(2.0);
  auto cm = g.add_operator(OperatorType::MATRIX_SCALE, {c, cm1});
  auto cm2 = g.add_operator(OperatorType::MATRIX_NEGATE, {cm});
  auto mlog1p = g.add_operator(OperatorType::MATRIX_LOG1MEXP, {cm2});

  // f(x) = log1mexp(-2 * g(x))
  // g(x) = [0.12, 0.34]
  // but we artificially set
  // g'(x) = [1.1, 1.1]
  // g''(x) = [2.3, 2.3]
  // for testing.

  Node* cm1_node = g.get_node(cm1);
  cm1_node->Grad1 = Eigen::MatrixXd::Ones(1, 2) * 1.1;
  auto g1 = cm1_node->Grad1.array();
  cm1_node->Grad2 = Eigen::MatrixXd::Ones(1, 2) * 2.3;
  auto g2 = cm1_node->Grad2.array();
  Node* cm_node = g.get_node(cm);
  Node* cm2_node = g.get_node(cm2);
  Node* mlog1p_node = g.get_node(mlog1p);
  std::mt19937 gen;
  cm_node->eval(gen);
  cm_node->compute_gradients();
  cm2_node->eval(gen);
  cm2_node->compute_gradients();
  mlog1p_node->eval(gen);
  mlog1p_node->compute_gradients();

  // For debugging, check the gradients of cm_node
  EXPECT_NEAR_MATRIX(-g0 * 2, cm2_node->value._matrix.array());
  EXPECT_NEAR_MATRIX(-g1 * 2, cm2_node->Grad1.array());
  EXPECT_NEAR_MATRIX(-g2 * 2, cm2_node->Grad2.array());

  // First derivative:
  auto first_grad = mlog1p_node->Grad1.array();
  // f(x) = log1mexp(-2 * g(x)) = log(1 - exp(-2 * g(x)))
  // f' = (2 g'(x))/(e^(2 g(x)) - 1) = f1top / f1bot
  // where f1top = 2 g'(x)
  //       f1bot = e^(2 g(x)) - 1
  auto f1top = 2 * g1;
  auto e2g = (2 * g0).exp();
  auto f1bot = e2g - 1;
  auto expected_f1 = f1top / f1bot;
  EXPECT_NEAR_MATRIX(expected_f1, first_grad);

  // f'' = (f1bot f1top' - f1top f1bot') / f1bot^2
  auto second_grad = mlog1p_node->Grad2.array();
  // where f1top' = 2 g''(x)
  //       f1bot' = 2 e^(2 g(x)) g'(x)
  auto f1top1 = 2 * g2;
  auto f1bot1 = 2 * e2g * g1;
  auto expected_f2 = (f1bot * f1top1 - f1top * f1bot1) / f1bot.pow(2);
  EXPECT_NEAR_MATRIX(expected_f2, second_grad);
}

TEST(testgradient, matrix_log1mexp_grad_backward) {
  /*
# Test backward differentiation
#
# Build the same model in PyTorch and BMG; we should get the same
# backwards gradients as PyTorch.

import torch
# torch doesn't yet implement log1mexp = log(1-exp(-|x|))
# see https://github.com/pytorch/pytorch/issues/39242
# See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
def log1mexp(x):
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)),
       torch.log1p(-torch.exp(x)))
hn = torch.distributions.HalfNormal(1)
hn_sample = torch.tensor([0.5, 5.2], requires_grad=True)
mlog_neg = log1mexp(hn_sample)
mlog = - mlog_neg
n = torch.distributions.Normal(mlog, 1.0)
ns = torch.tensor([2.5, 0.15], requires_grad=True)
log_prob = n.log_prob(ns).sum() + hn.log_prob(hn_sample).sum()
print(torch.autograd.grad(log_prob, hn_sample, retain_graph=True))
# [ 4.7916, -5.1991]
print(torch.autograd.grad(log_prob, ns, retain_graph=True))
# [-3.4328, -0.1555]

  */

  Graph g;
  auto one = g.add_constant_pos_real(1.0);
  auto hn = g.add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::POS_REAL, {one});
  auto two = g.add_constant_natural(2);
  // sample 0
  auto hn_sample =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{hn, two});
  Eigen::MatrixXd hn_observed(2, 1);
  hn_observed << 0.5, 5.2;
  g.observe(hn_sample, hn_observed);
  auto hn_sample_neg = g.add_operator(OperatorType::MATRIX_NEGATE, {hn_sample});

  // MATRIX_LOG1MEXP takes negative values and returns negative values.
  auto mlog_neg =
      g.add_operator(OperatorType::MATRIX_LOG1MEXP, {hn_sample_neg});
  auto mlog = g.add_operator(OperatorType::TO_REAL_MATRIX, {mlog_neg});

  auto index_zero = g.add_constant_natural(0);
  auto mlog0 = g.add_operator(OperatorType::INDEX, {mlog, index_zero});
  auto n0 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog0, one});
  // sample 1
  auto ns0 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n0});
  g.observe(ns0, 2.5);

  auto index_one = g.add_constant_natural(1);
  auto mlog1 = g.add_operator(OperatorType::INDEX, {mlog, index_one});
  auto n1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mlog1, one});
  // sample 2
  auto ns1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n1});
  g.observe(ns1, 0.15);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), 4.7916, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), -5.1991, 1e-3);
  EXPECT_NEAR((*grad1[1]), -3.4328, 1e-3);
  EXPECT_NEAR((*grad1[2]), -0.1555, 1e-3);
}

TEST(testgradient, matrix_complement_grad_forward) {
  Graph g;

  // Test forward differentiation

  Eigen::MatrixXd x0(1, 2);
  x0 << 0.15, 0.77;
  auto x = g.add_constant_probability_matrix(x0);
  Node* x_node = g.get_node(x);

  Eigen::MatrixXd x1(1, 2);
  x1 << 1.1, 2.2;
  x_node->Grad1 = x1;

  Eigen::MatrixXd x2(1, 2);
  x2 << 3.4, 4.5;
  x_node->Grad2 = x2;

  auto xc = g.add_operator(OperatorType::MATRIX_COMPLEMENT, {x});
  Node* xc_node = g.get_node(xc);

  std::mt19937 gen;
  xc_node->eval(gen);
  xc_node->compute_gradients();

  // f(x) = 1 - g(x)
  // g(x) = [0.15, 0.77]
  // but we artificially set
  // g'(x) = [1.1, 2.2]
  // g''(x) = [3.4, 4.5]
  // for testing.

  Eigen::MatrixXd g1 = xc_node->Grad1;
  Eigen::MatrixXd g2 = xc_node->Grad2;

  Eigen::MatrixXd m1s(1, 2);
  m1s << -1, -1;
  auto expected_g1 = m1s.array() * x1.array();
  auto expected_g2 = m1s.array() * x2.array();

  EXPECT_NEAR_MATRIX(g1, expected_g1);
  EXPECT_NEAR_MATRIX(g2, expected_g2);
}

TEST(testgradient, matrix_complement_grad_backward) {
  /*
# Test backward differentiation
#
# Build the same model in PyTorch and BMG; we should get the same
# backwards gradients as PyTorch.

import torch
beta = torch.distributions.Beta(2, 2)
beta_sample = torch.tensor([0.1, 0.7], requires_grad=True)
complement = 1 - beta_sample
n = torch.distributions.Normal(complement, 1.0)
ns = torch.tensor([1.1, 2.2], requires_grad=True)
log_prob = beta.log_prob(beta_sample).sum() + n.log_prob(ns).sum()
print(torch.autograd.grad(log_prob, beta_sample, retain_graph=True))
# [ 8.6889, -3.8048]
print(torch.autograd.grad(log_prob, ns, retain_graph=True))
# [-0.2000, -1.9000]

  */

  Graph g;
  auto one = g.add_constant_pos_real(1.0);
  auto two = g.add_constant_pos_real(2.0);
  auto beta = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, {two, two});

  // sample 0
  auto beta_sample = g.add_operator(
      OperatorType::IID_SAMPLE,
      std::vector<uint>{beta, g.add_constant_natural(2)});
  Eigen::MatrixXd beta_observed(2, 1);
  beta_observed << 0.1, 0.7;
  g.observe(beta_sample, beta_observed);

  auto complement =
      g.add_operator(OperatorType::MATRIX_COMPLEMENT, {beta_sample});

  // COMPLEMENT takes probability values and returns probability values.
  auto comp = g.add_operator(OperatorType::TO_REAL_MATRIX, {complement});

  auto index_zero = g.add_constant_natural(0);
  auto comp0 = g.add_operator(OperatorType::INDEX, {comp, index_zero});
  auto n0 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {comp0, one});
  // sample 1
  auto ns0 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n0});
  g.observe(ns0, 1.1);

  auto index_one = g.add_constant_natural(1);
  auto comp1 = g.add_operator(OperatorType::INDEX, {comp, index_one});
  auto n1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {comp1, one});
  // sample 2
  auto ns1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{n1});
  g.observe(ns1, 2.2);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), 8.6889, 1e-3);
  EXPECT_NEAR((*grad1[0])(1), -3.8048, 1e-3);
  EXPECT_NEAR((*grad1[1]), -0.2000, 1e-3);
  EXPECT_NEAR((*grad1[2]), -1.9000, 1e-3);
}
