// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

/// TODO[Walid]: Is it essential that we test distributions using graphs?
using namespace beanmachine::graph;

/// Tests with scalar samples
TEST(testdistrib, half_normal) {
  Graph g;
  /// TODO[Walid]: Parameterize expected values by these variables
  /// TODO[Walid]: Move declarations and graph additions closer to use
  const double MEAN = 0; /// Half-normal assumes 0
  const double STD = 3.0;
  auto real1 = g.add_constant(MEAN);
  auto pos1 = g.add_constant_pos_real(STD);
  /// TODO[Walid]: Make argument be POS_REAL rather than REAL
  /// Check that g.add_distribution checks arguments to HALF_NORMAL correctly
  // negative tests half_normal has one parent
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{pos1, pos1}),
      std::invalid_argument);
  // negative test the parent must be a positive real
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1}),
      std::invalid_argument);
  // test creation of a distribution
  auto half_normal_dist = g.add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::REAL, std::vector<uint>{pos1});
  // test distribution of mean and variance.
  /// The following line is adding the following declaration to the graph:
  /// real_val   \in Half_Normal (m,s)
  /// which, basically defines an f(x) = Half_Normal(m,s,x)
  auto real_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_normal_dist});
  auto real_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  /// Since we are not making observations, the following is just
  /// what we expect from directly sampling from distribution
  EXPECT_NEAR(means[0], STD * std::sqrt(2.0 / M_PI), 0.1);
  EXPECT_NEAR(
      means[1] - means[0] * means[0], STD * STD * (1.0 - 2.0 / M_PI), 0.1);
  // test log_prob
  /// The following just tests log prob at 1.0. It has no connection to above
  /// code that uses "infer"
  ///
  /// Set the value of real_val to 1.0
  /// The log(prob(real_val)) is therefore -ln(3)-0.5*ln(pi/2)-0.5*(1/3)^2
  /// = -1.37995919687
  g.observe(real_val, 1.0);
  EXPECT_NEAR(
      g.log_prob(real_val), -1.37995919687, 0.001); /// computed by hand!

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
  /// = - (1 - 0) / 9 + -2 * (3 - 1) /9
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
  /// all the distribution codes in our codebase.

  EXPECT_NEAR(real_sq_val, 4.0, 0.01); /// Just the index value!
  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL, /// TODO[Walid]: Consider half_normal here too
      AtomicType::REAL,
      std::vector<uint>{real_sq_val, pos1});
  auto real_val2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist2});
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
  auto normal_dist3 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos_sq_val});
  auto real_val3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist3});
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
      std::vector<uint>{pos_one});
  uint mu =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_normal_dist});

  uint dist_y = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{pos_one}); // TODO[Walid]: Replacing pos_one by mu fails
  uint y =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist_y, two});
  g.observe(mu, 0.0); // TODO[Walid]: Would need to be non-zero to serve as s
  Eigen::MatrixXd yobs(2, 1);
  yobs << 0.5, 1.5;
  g.observe(y, yobs);

  // test log_prob() on vector value:

  /// Recall:
  ///    Half_Normal(sigma,x)
  ///         = 1/(sigma sqrt(pi/2)) *  exp(-0.5*(x/sigma)^2)
  ///    log(f(x|s))
  ///         = logprob  = -log(s) - 0.5 * log(pi/2) - 0.5 * (x/s)^2
  /// Now consider the model:
  ///   mu ~ Half_Normal(1)
  ///   y  ~ Half_Normal(1)^2
  /// Under the observations mu=0.0 and y=[0.5 1.5]
  /// What is log_prob(y)?
  /// That means we want to compute log(f(y))
  /// TODO[Walid]: At some point, find a way to make y depend on mu above
  /// log(f(y)) = -ln(1) -ln(pi/2) - 0.5((-0.5)^2 + (-1.5)^2)
  /// = -ln(pi/2) - 0.5*(0.5^2+1.5^2)
  /// = -1.70158270529 // log_prob_y below
  double log_prob_y = g.log_prob(y);
  EXPECT_NEAR(log_prob_y, -1.70158270529, 0.001);

  // test backward_param(), backward_value() and
  // backward_param_iid(), backward_value_iid():

  /// Recall that for Half Normal:
  ///  sample   = N(-log(s) - 0.5 * log(pi/2))
  ///          -0.5 * squares/s^2

  /// In contrast to working with f(y|mu) above
  /// here we will work with f(y,mu) = f(y)*f(mu)
  /// For a y=(y1,y2) the log prob is given by
  /// log(f(y))
  /// = -(2/2)ln(pi/2) - 0.5((-y1)^2 + (-y2)^2)/s^2
  /// and so derivative wrt mu is:
  /// = 0.0 in our case /// grad1[0] below
  /// and derivative wrt y1 is:
  /// = - y1
  /// = -0.5 in our case ///grad1[1]->1 below
  /// and derivative wrt y2 is:
  /// = - y2
  /// = -1.5 in our case ///grad1[1]->2 below

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_double, -0.0, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(0), -0.5, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(1), -1.5, 1e-3);

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
      std::vector<uint>{s}); // TODO[Walid]: Mean was m1. Introduce an s1
  auto d2 = g2.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{s}); // TODO[Walid]: Mean was m2. Introduce an s2
  auto p = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      std::vector<uint>{p, d1, d2});
  auto x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto xiid =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(m1, 0.0);
  g2.observe(m2, 0.0);
  g2.observe(s, 1.8);
  g2.observe(p, 0.37);
  g2.observe(x, 0.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, 1.5;
  g2.observe(xiid, xobs);
  /// First, the checking the log-probabilities
  /// The following Python code spells out the
  /// calculation we use for checking the full
  /// log prob. Note that all FLAT dists contribute
  /// 0 to the total log prob.
  /// from numpy import pi,log,exp
  /// s = 1.8
  /// p = 0.37
  /// x = 0.5
  /// x1 = 0.5
  /// x2 = 1.5
  /// def log_probability(x):
  ///     ## Logprob for first Normal
  ///     lp_d1 = - log(s) - 0.5 * log(pi/2) - 0.5 * (x/ s)**2
  ///     ## Logprob for second Normal
  ///     lp_d2 = - log(s) - 0.5 * log(pi/2) - 0.5 * (x/ s)**2
  ///     ## The mixture part
  ///     q = p * exp(lp_d1) + (1-p)* exp(lp_d2)
  ///     lp_q = log(q)
  ///     return lp_q
  /// sum([log_probability(x) for x in [x,x1,x2]])
  /// ## == -2.865116768689922

  EXPECT_NEAR(g2.full_log_prob(), -2.865116768689922, 1e-3);
  ///
  /// s = torch.tensor(1.8, requires_grad=True)
  /// p = torch.tensor(0.37, requires_grad=True)
  /// x = torch.tensor([0.5, 0.5, 1.5], requires_grad=True)
  /// d1 = torch.distributions.Normal(0, s)
  /// d2 = torch.distributions.Normal(0, s)
  /// f1 = d1.log_prob(x).exp()
  /// f2 = d2.log_prob(x).exp()
  /// log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  /// torch.autograd.grad(log_p, x)[0]
  /// The calculations above simply return the derivatives
  /// of the value sum([log_probability...]) defined above
  /// with respect to various variables. For example, the
  /// derivative wrt to x would be can be computed by the
  /// following function:
  /// def log_probability_x(x):
  ///     ## First Normal
  ///     lp_d1 = - log(s) - 0.5 * log(2*pi) - 0.5 * (x/ s)**2
  ///     lp_d1_x = - x/ s**2
  ///     ## Second Normal
  ///     lp_d2 = - log(s) - 0.5 * log(2*pi) - 0.5 * (x/ s)**2
  ///     lp_d2_x = - x/ s**2
  ///     ## The mixture part
  ///     q = p * exp(lp_d1) + (1-p)* exp(lp_d2)
  ///     q_x = p * exp(lp_d1) * lp_d1_x + (1-p)* exp(lp_d2)*lp_d2_x
  ///     lp_q = log(q)
  ///     lp_q_x = (1/q)*q_x
  ///     return lp_x_x
  /// [log_probability_x(x) for x in [x,x1,x2]]
  /// ## [-0.154320987654321, -0.154320987654321, -0.4629629629629629]
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 6);
  EXPECT_NEAR(back_grad[0]->_double, 0.0, 1e-3); // m1 /// dummy param
  EXPECT_NEAR(back_grad[1]->_double, 0.0, 1e-3); // m2 /// dummy param
  EXPECT_NEAR(back_grad[2]->_double, -1.1951, 1e-3); // s
  EXPECT_NEAR(back_grad[3]->_double, 0.0, 1e-3); // p
  EXPECT_NEAR(back_grad[4]->_double, -0.1543, 1e-3); // x
  EXPECT_NEAR(back_grad[5]->_matrix.coeff(0), -0.1543, 1e-3); // xiid
  EXPECT_NEAR(back_grad[5]->_matrix.coeff(1), -0.4630, 1e-3);
}
