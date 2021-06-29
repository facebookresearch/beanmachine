// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

/// TODO[Walid]: Is it essential that we test distributions using graphs?
using namespace beanmachine::graph;

/// Tests with scalar samples
TEST(testdistrib, half_normal) {
  Graph g;
  /// TODO[Walid]: Would be good to parameterize later tests by these values!
  const double MEAN = 0; /// TODO[Walid]: Half-normal assumes 0
  const double STD = 3.0;
  auto real1 = g.add_constant(MEAN);
  auto pos1 = g.add_constant_pos_real(STD);
  /// TODO[Walid]: Argument will become just one, and value should be POS_REAL
  /// Check that g.add_distribution checks arguments to HALF_NORMAL correctly
  // negative tests half_normal has two parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1, pos1, real1}),
      std::invalid_argument);
  // negative test the parents must be a real and a positive
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1, real1}),
      std::invalid_argument);
  // test creation of a distribution
  auto half_normal_dist = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});
  // test distribution of mean and variance.
  /// The following line is adding the following declaration to the graph:
  /// real_val   \in Normal (m,s)
  /// which, basically defines an f(x) = Normal(m,s,x)
  auto real_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_normal_dist});
  auto real_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  /// TODO[Walid]: Following should be OK as we change MEAN and STD...
  /// In particular, we are not making observation, so, this just
  /// samples directly from distribution
  /// TODO[Walid]: Following should be OK as we change distributions!
  EXPECT_NEAR(means[0], MEAN, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], STD * STD, 0.1);
  // test log_prob
  /// The following just tests log prob at 1.0. It has no connection to above
  /// code that uses "infer"
  /// TODO[Walid]: First update the following to work for a MEAN of 0.0, and
  /// then test again after the MEAN parameter and fields have been removed.
  ///
  /// Set the value of real_val to 1.0
  /// The log(prob(real_val)) is therefore -ln(3)-0.5*ln(2*pi)-0.5*(1/3)^2
  /// = -2.07310637743
  g.observe(real_val, 1.0);
  EXPECT_NEAR(
      g.log_prob(real_val), -2.07310637743, 0.001); /// computed by hand!

  // test gradient of log_prob w.r.t. value and the mean

  /// First, a simple check on a single distribution
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(real_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.1111, 0.01); /// By hand, - (x - m) / s^2 = -1/9
  EXPECT_NEAR(grad2, -0.1111, 0.01); /// By hand, - 1 / s^2 = -1/9
  /// Second, a check on a composite model

  /// Recall that:
  /// real_val   \in Normal (m,s)          -- defines f(x) = Normal(m,s,x)
  /// and also define
  /// real_val_2 \in Normal (real_val^2,s) -- defines f(y|x) = Normal(x^2,s,y)
  ///    also, we observed that real_value = 1
  /// Note that where as new definitions incrementally add definitions for
  /// probabilities conditioned on other variales, at the end we are interested
  /// in the joint probablility distribution (and it's logprob, logprob
  /// gradients, etc) So, for our example, we want to compute log(f(y,x))'[x].
  /// Terminology: joint distribution means f(y,x) = f(y|x)*f(x)
  /// Recall also that log(Normal(m,s,x))'[x] = - (x - m) / s^2 call this (*).
  /// Recall also that
  ///     log(Normal(m,s,x))
  ///   = - log(s) -0.5 log(2*pi) - 0.5 (x - m)^2 / s^2
  ///   and call this (**)
  /// Let's work this out by hand to see what we should expect:
  ///    log(f(y,x))'[x]                   /// f(y,x) = f(y|x) * f(x)
  ///  = log(f(y|x)*f(x))'[x]              /// log(a*b) = log(a) + log(b)
  ///  = (log(f(y|x)+log(f(x)))'[x]        /// (a+b)'[x] = a'[x] + b'[x]
  ///  = log(f(y|x))'[x] + log(f(x))'[x]   /// a+b = b + a
  ///  = log(f(x))'[x] + log(f(y|x))'[x]   /// (*)
  ///  = - (x - m) / s^2 + log(f(y|x))'[x] /// f(y|x) definition
  ///  = - (x - m) / s^2 + log(Normal(x^2,s,y))'[x] /// (**) for some k
  ///  = - (x - m) / s^2 + (k - 0.5 (y - x^2)^2 / s^2 )'[x] /// k'[...] = 0
  ///  = - (x - m) / s^2 + (0.5 (y - x^2)^2 / s^2 )'[x]
  /// /// (y-x^2)'[x]=-2x and (y-x^2)^2'[y-x^2]=2(y-x^2)
  /// = - (x - m) / s^2 + -2x * (y - x^2) / s^2 /// m=0,s=3,x=1,y=3
  /// = - (1 + 0) / 9 + -2 * (3 - 1) /9
  /// = -1 / 9 + 4 / 9 = 3/9 /// This will be grad1 below
  ///
  /// For the calculation of the second derivative, we can reuse the above
  /// before the substitution, that is:
  ///    log(f(y,x))''[x]
  ///  = (- (x - m) / s^2 + -2x * (y - x^2) / s^2)'[x]
  ///  = - 1 / s^2 + (-2x * (y - x^2) / s^2)'[x]
  ///  = - 1 / s^2 - 2 * ((x y - x^3) / s^2)'[x]
  ///  = - 1 / s^2 - 2 * (y - 3x^2) / s^2 /// m=0,s=3,x=1,y=3
  ///  = - 1 / 9   - 2 * (3 - 3) / s^2
  ///  = -1/9 /// This will be grad2 below
  ///

  /// Note: Because we have observed/fixed real_val, and because we
  /// specify in the graph that real_sq_val = real_val * real_val,
  /// and because log_prob traverse the entire graph and propagates
  /// such constraints, we actual have a fixed value for real_sq_val
  /// as we compute log-prog.

  /// TODO[Lily]: One thing that makes this code harder to read is
  /// interspersing setting of values with inference and with tests,
  /// and to help the reader it is good to have comments such as the
  /// ones above, and pointing out the significance of setting some
  /// observations as we go along (and also setting up some relations
  /// in the graph). TODO[Walid]: It would be good to do this for
  /// all the distribution codes in our codebase

  EXPECT_NEAR(real_sq_val, 4.0, 0.01); /// Just the index value!
  auto half_normal_dist2 = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real_sq_val, pos1});
  auto real_val2 = g.add_operator(
      OperatorType::SAMPLE, std::vector<uint>{half_normal_dist2});
  g.observe(real_val2, 3.0);
  grad1 = 0;
  grad2 = 0;
  g.gradient_log_prob(real_val, grad1, grad2);
  EXPECT_NEAR(grad1, 0.33333, 0.01);
  EXPECT_NEAR(grad2, -0.11111, 0.01);
  ///
  // test gradient of log_prob w.r.t. sigma
  ///
  /// The problem that we will consider is basically:
  /// pos_val ~ Half_Cauchy(3.5,pos_val)  --- f(x) = Half_Cauchy(s,x),  s=3.5
  /// real_val3 ~ Normal(real1,pos_val^2) --- f(y|x) = Normal(m,x^2,y), m=0
  /// As observations, we take pos_val =   x = 7
  ///                      and real_val3 = y = 5.0
  /// We want to compute (log(f(y,x))'[x] and ''[x].
  /// So here it goes:
  /// (log(f(y,x))'[x] = (log(f(y|x)*f(x)))'[x] = (log(f(y|x)) + log(f(x)))'[x]
  /// = (log(Normal(m,x^2,y)) + log(Half_Cauchy(s,x)))'[x] /// Call this (*)
  /// Recall:
  ///    log(Normal(m,s,x))   = -log(s  ) -0.5 log(2*pi) - 0.5 (x - m)^2 / s^2
  /// so log(Normal(m,x^2,y)) = -log(x^2) -0.5 log(2*pi) - 0.5 (y - m)^2 / x^4
  /// Recall:
  ///    log(Half_Cauchy(s,x))= -log(pi/2) -log(s) -log(1 + (x/s)^2)
  /// Continuing with (*)
  /// = (-log(x^2) -0.5 log(2*pi) - 0.5 (y - m)^2 / x^4 +
  ///   + -log(pi/2) -log(s) -log(1 + (x/s)^2))'[x]
  /// = (-2x/x^2 + 4 * 0.5 (y - m)^2 / x^5 +
  ///   -(2x/s^2)/(1 + (x/s)^2))
  /// = (-2/x + 2 (y - m)^2 / x^5 +
  ///   -(2x)/(s^2 + x^2))
  /// = -2/7 + 2 (5 + 0)^2 / 7^5 -14/(3.5^2 + 7^2)
  /// = -0.51131076337 /// grad1 below
  ///
  /// (log(_))''[x] = (-2/x + 2 (y - m)^2 / x^5 -(2x)/(s^2 + x^2))'[x]
  /// = 2/x^2 - 10 (y - m)^2 / x^6 + (2x^2 - 2x^2)/(s^2+x^2)^2
  /// = 2/7^2 - 10 (5 + 0)^2 / 7^6 + (2*7^2 - 2*3.5^2)/(3.5^2+7^2)^2
  /// = 0.05828319832 /// grad2 below

  const double SCALE = 3.5;
  auto pos_scale = g.add_constant_pos_real(SCALE);
  auto half_cauchy_dist = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos_scale});
  auto pos_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_cauchy_dist});
  g.observe(pos_val, 7.0);
  auto pos_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{pos_val, pos_val});
  auto half_normal_dist3 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos_sq_val});
  auto real_val3 = g.add_operator(
      OperatorType::SAMPLE, std::vector<uint>{half_normal_dist3});
  g.observe(real_val3, 5.0);
  grad1 = grad2 = 0;
  g.gradient_log_prob(pos_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.51131076337, 1e-6);
  EXPECT_NEAR(grad2, 0.05828319832, 1e-6);
}

/// Tests with aggregate samples
TEST(testdistrib, backward_half_normal_half_normal) {
  Graph g;
  uint zero = g.add_constant(0.0);
  uint pos_one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant((natural_t)2);

  uint half_normal_dist = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos_one});
  uint mu =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_normal_dist});

  uint dist_y = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{mu, pos_one});
  uint y =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist_y, two});
  g.observe(mu, 0.1);
  Eigen::MatrixXd yobs(2, 1);
  yobs << 0.5, -0.5;
  g.observe(y, yobs);

  // test log_prob() on vector value:

  /// Recall:
  ///    Normal(mu,sigma,x) = 1/(sigma sqrt(2pi)) *  exp(-0.5*((x-mu)/sigma)^2)
  /// Now consider the model:
  ///   mu ~ Normal(0,1)
  ///   y  ~ Normal(mu,1)^2
  /// Under the observations mu=0.1 and y=[0.5 -0.5]
  /// What is log_prob(y)?
  /// That means we want to compute log(f(y|mu))
  /// Note: In particular, here we do NOT compute log(f(y,mu))!
  /// log(f(y|mu)) = -ln(2pi) - 0.5((mu-0.5)^2 + (mu+0.5)^2)
  /// = -ln(2*pi) - 0.5*(0.4^2+0.6^2)
  /// = -2.09787706641 // log_prob_y below
  double log_prob_y = g.log_prob(y);
  EXPECT_NEAR(log_prob_y, -2.0979, 0.001);

  // test backward_param(), backward_value() and
  // backward_param_iid(), backward_value_iid():

  /// In contrast to working with f(y|mu) above
  /// here we will work with f(y,mu) = f(y|mu)*f(mu)
  /// For a y=(y1,y2) the log prob is given by
  /// log(f(y,mu))
  /// = -(3/2)ln(2 pi) - 0.5(mu^2 + (mu-y1)^2 + (mu-y2)^2)
  /// and so derivative wrt mu is:
  /// -(3mu - y1 - y2)
  /// = -0.3 in our case /// grad1[0] below
  /// and derivative wrt y1 is:
  /// = mu - y1
  /// = -0.4 in our case ///grad1[1]->1 below
  /// and derivative wrt y2 is:
  /// = mu - y2
  /// = 0.6 in our case ///grad1[1]->2 below

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_double, -0.3, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(0), -0.4, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(1), 0.6, 1e-3);

  // mixture of half_normals
  /// Checking the log probability and the
  /// back gradient calculations with respect to several
  /// sampled variables, and through a composition of distributions.
  Graph g2;
  auto size = g2.add_constant((natural_t)2);
  auto flat_real = g2.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto flat_pos = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto m1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto m2 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto s = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto d1 = g2.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{m1, s});
  auto d2 = g2.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{m2, s});
  auto p = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      std::vector<uint>{p, d1, d2});
  auto x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto xiid =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(m1, -1.2);
  g2.observe(m2, 0.4);
  g2.observe(s, 1.8);
  g2.observe(p, 0.37);
  g2.observe(x, -0.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, -1.5;
  g2.observe(xiid, xobs);
  /// First, the checking the log-probabilities
  /// The following Python code spells out the
  /// calculation we use for checking the full
  /// log prob. Note that all FLAT dists contribute
  /// 0 to the total log prob.
  /// from numpy import pi,log,exp
  /// m1 = -1.2
  /// m2 = 0.4
  /// s = 1.8
  /// p = 0.37
  /// x = -0.5
  /// x1 = 0.5
  /// x2 = -1.5
  // def log_probability(x):
  ///     ## Logprob for first Normal
  ///     lp_d1 = - log(s) - 0.5 * log(2*pi) - 0.5 * ((x-m1)/ s)**2
  ///     ## Logprob for second Normal
  ///     lp_d2 = - log(s) - 0.5 * log(2*pi) - 0.5 * ((x-m2)/ s)**2
  ///     ## The mixture part
  ///     q = p * exp(lp_d1) + (1-p)* exp(lp_d2)
  ///     lp_x = log(q)
  ///     return lp_x
  /// sum([log_probability(x) for x in [x,x1,x2]])
  /// ## == -5.091080467031949
  EXPECT_NEAR(g2.full_log_prob(), -5.0911, 1e-3);
  ///
  // To verify the derivatives with pyTorch:
  // m1 = torch.tensor(-1.2, requires_grad=True)
  // m2 = torch.tensor(0.4, requires_grad=True)
  // s = torch.tensor(1.8, requires_grad=True)
  // p = torch.tensor(0.37, requires_grad=True)
  // x = torch.tensor([-0.5, 0.5, -1.5], requires_grad=True)
  // d1 = torch.distributions.Half_Normal(m1, s)
  // d2 = torch.distributions.Half_Normal(m2, s)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(log_p, x)[0]
  /// The calculations above simply return the derivatives
  /// of the value sum([log_probability...]) defined above
  /// with respect to various variables. For example, the
  /// derivative wrt to m1 would be can be computed by the
  /// following function:
  /// def log_probability_m1(x):
  ///     ## First Normal
  ///     lp_d1 = - log(s) - 0.5 * log(2*pi) - 0.5 * ((x-m1)/ s)**2
  ///     lp_d1_m1 = (x-m1) / (s**2)
  ///     ## Second Normal
  ///     lp_d2 = - log(s) - 0.5 * log(2*pi) - 0.5 * ((x-m2)/ s)**2
  ///     lp_d2_m1 = 0
  ///     ## The mixture part
  ///     q = p * exp(lp_d1) + (1-p)* exp(lp_d2)
  ///     q_m1 = p * exp(lp_d1) * lp_d1_m1
  ///     lp_x = log(q)
  ///     lp_x_m1 = (1/q)*q_m1
  ///     return lp_x_m1
  /// sum([log_probability_m1(x) for x in [x,x1,x2]]) ## 0.17942185552302908
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 6);
  EXPECT_NEAR(back_grad[0]->_double, 0.1794, 1e-3); // m1
  EXPECT_NEAR(back_grad[1]->_double, -0.4410, 1e-3); // m2
  EXPECT_NEAR(back_grad[2]->_double, -1.0964, 1e-3); // s
  EXPECT_NEAR(back_grad[3]->_double, 0.2054, 1e-3); // p
  EXPECT_NEAR(back_grad[4]->_double, 0.0893, 1e-3); // x
  EXPECT_NEAR(back_grad[5]->_matrix.coeff(0), -0.1660, 1e-3); // xiid
  EXPECT_NEAR(back_grad[5]->_matrix.coeff(1), 0.3381, 1e-3);
}
