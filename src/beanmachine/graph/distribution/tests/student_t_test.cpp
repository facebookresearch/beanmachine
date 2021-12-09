/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, student_t) {
  Graph g;
  const double DOF = 3;
  const double LOC = -11.0;
  const double SCALE = 4.0;
  auto dof = g.add_constant_pos_real(DOF);
  auto loc = g.add_constant(LOC);
  auto scale = g.add_constant_pos_real(SCALE);
  // negative tests: student-t has three parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::STUDENT_T, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::STUDENT_T,
          AtomicType::REAL,
          std::vector<uint>{dof}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::STUDENT_T,
          AtomicType::REAL,
          std::vector<uint>{dof, loc}),
      std::invalid_argument);
  // negative test the parents must be a positive, real and a positive
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::STUDENT_T,
          AtomicType::REAL,
          std::vector<uint>{loc, dof, scale}),
      std::invalid_argument);
  // negative test: the sample type must be REAL
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::STUDENT_T,
          AtomicType::POS_REAL,
          std::vector<uint>{dof, loc, scale}),
      std::invalid_argument);
  // test creation of a distribution
  auto student_t_dist = g.add_distribution(
      DistributionType::STUDENT_T,
      AtomicType::REAL,
      std::vector<uint>{dof, loc, scale});
  // test distribution of mean and variance
  auto real_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{student_t_dist});
  auto real_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], LOC, 0.1);
  EXPECT_NEAR(
      means[1] - means[0] * means[0], (DOF / (DOF - 2)) * SCALE * SCALE, 1.0);
  // test log_prob
  // torch.distributions.StudentT(3, -11, 4).log_prob(-5) -> -3.5064
  g.observe(real_val, -5.0);
  EXPECT_NEAR(g.log_prob(real_val), -3.5064, 0.001);
  // test gradient of the sampled value
  // Verified in pytorch using the following code:
  // x = torch.tensor([-5.0], requires_grad=True)
  // f_x = torch.distributions.StudentT(3, -11, 4).log_prob(x)
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> -0.2857 and f_grad2 -> -0.0068
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(real_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.2857, 0.001);
  EXPECT_NEAR(grad2, -0.0068, 0.001);
  // test gradients of the parameters
  // dof ~ FLAT
  // loc ~ FLAT
  // scale ~ FLAT
  // x ~ StudentT(dof^2, loc^2, scale^2)
  dof = g.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  loc = g.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g.add_distribution(
          DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{})});
  scale = g.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto dof_sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{dof, dof});
  auto loc_sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{loc, loc});
  auto scale_sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{scale, scale});
  auto x_dist = g.add_distribution(
      DistributionType::STUDENT_T,
      AtomicType::REAL,
      std::vector<uint>{dof_sq, loc_sq, scale_sq});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{x_dist});
  g.observe(dof, 5.0);
  g.observe(loc, 2.5);
  g.observe(scale, 1.5);
  g.observe(x, 9.0);
  // dof = torch.tensor([5.0], requires_grad=True)
  // loc = torch.tensor([2.5], requires_grad=True)
  // scale = torch.tensor([1.5], requires_grad=True)
  // f_x = torch.distributions.StudentT(dof**2, loc**2, scale**2).log_prob(9.0)
  // f_grad_dof = torch.autograd.grad(f_x, dof, create_graph=True)
  // f_grad2_dof = torch.autograd.grad(f_grad_dof, dof)
  // f_grad_dof, f_grad2_dof -> 0.0070, -0.0042
  // f_grad_loc = torch.autograd.grad(f_x, loc, create_graph=True)
  // f_grad2_loc = torch.autograd.grad(f_grad_loc, loc)
  // f_grad_loc, f_grad2_loc -> 2.6654 , -3.2336
  // f_grad_scale = torch.autograd.grad(f_x, scale, create_graph=True)
  // f_grad2_scale = torch.autograd.grad(f_grad_scale, scale)
  // f_grad_scale, f_grad2_scale -> 0.6213, -5.3327
  // we will call gradient_log_prob from all the parameters one time
  // to ensure that their children are evaluated
  g.gradient_log_prob(loc, grad1, grad2);
  g.gradient_log_prob(scale, grad1, grad2);
  // test dof
  grad1 = grad2 = 0;
  g.gradient_log_prob(dof, grad1, grad2);
  EXPECT_NEAR(grad1, 0.0070, 0.001);
  EXPECT_NEAR(grad2, -0.0042, 0.001);
  // test loc
  grad1 = grad2 = 0;
  g.gradient_log_prob(loc, grad1, grad2);
  EXPECT_NEAR(grad1, 2.6654, 0.001);
  EXPECT_NEAR(grad2, -3.2336, 0.001);
  // test scale
  grad1 = grad2 = 0;
  g.gradient_log_prob(scale, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6213, 0.001);
  EXPECT_NEAR(grad2, -5.3327, 0.001);

  // test backward_param, backward_value:
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 5);
  EXPECT_NEAR(grad[0]->_double, -0.2857, 1e-3); // real_val
  EXPECT_NEAR(grad[1]->_double, 0.0070, 1e-3); // dof
  EXPECT_NEAR(grad[2]->_double, 2.6654, 1e-3); // loc
  EXPECT_NEAR(grad[3]->_double, 0.6213, 1e-3); // scale

  // test log_prob, backward_param_iid, backward_value_iid:
  auto two = g.add_constant((natural_t)2);
  auto y = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{x_dist, two, two});
  Eigen::MatrixXd m(2, 2);
  m << 7.1, 2.6, 5.8, 12.2;
  g.observe(y, m);
  EXPECT_NEAR(g.log_prob(dof), -14.0561, 1e-3);
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 6);
  EXPECT_NEAR(grad[1]->_double, -0.0774, 1e-3); // dof
  EXPECT_NEAR(grad[2]->_double, 4.4557, 1e-3); // loc
  EXPECT_NEAR(grad[3]->_double, 6.4192, 1e-3); // scale
  EXPECT_NEAR(grad[4]->_double, -0.5331, 1e-3); // x
  EXPECT_NEAR(grad[5]->_matrix.coeff(0), -0.1736, 1e-3); // y
  EXPECT_NEAR(grad[5]->_matrix.coeff(1), 0.0923, 1e-3);
  EXPECT_NEAR(grad[5]->_matrix.coeff(2), 0.6784, 1e-3);
  EXPECT_NEAR(grad[5]->_matrix.coeff(3), -0.9551, 1e-3);

  // test sample/iid_sample from a mixture of t-dist
  Graph g2;
  auto size = g2.add_constant((natural_t)2);
  auto flat_real = g2.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto flat_pos = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto df = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto loc1 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto loc2 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto s = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto d1 = g2.add_distribution(
      DistributionType::STUDENT_T,
      AtomicType::REAL,
      std::vector<uint>{df, loc1, s});
  auto d2 = g2.add_distribution(
      DistributionType::STUDENT_T,
      AtomicType::REAL,
      std::vector<uint>{df, loc2, s});
  auto p = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      std::vector<uint>{p, d1, d2});
  auto x1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto x2 =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(df, 10.0);
  g2.observe(loc1, -1.2);
  g2.observe(loc2, 0.4);
  g2.observe(s, 1.8);
  g2.observe(p, 0.37);
  g2.observe(x1, -0.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, -1.5;
  g2.observe(x2, xobs);
  // To verify the results with pyTorch:
  // df = torch.tensor(10.0, requires_grad=True)
  // l1 = torch.tensor(-1.2, requires_grad=True)
  // l2 = torch.tensor(0.4, requires_grad=True)
  // s = torch.tensor(1.8, requires_grad=True)
  // p = torch.tensor(0.37, requires_grad=True)
  // x = torch.tensor([-0.5, 0.5, -1.5], requires_grad=True)
  // d1 = torch.distributions.StudentT(df, l1, s)
  // d2 = torch.distributions.StudentT(df, l2, s)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // f_x = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(f_x, x)[0]
  EXPECT_NEAR(g2.full_log_prob(), -5.1944, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 7);
  EXPECT_NEAR(back_grad[0]->_double, 0.0102, 1e-3); // df
  EXPECT_NEAR(back_grad[1]->_double, 0.1804, 1e-3); // loc1
  EXPECT_NEAR(back_grad[2]->_double, -0.4446, 1e-3); // loc2
  EXPECT_NEAR(back_grad[3]->_double, -1.0941, 1e-3); // s
  EXPECT_NEAR(back_grad[4]->_double, 0.2134, 1e-3); // p
  EXPECT_NEAR(back_grad[5]->_double, 0.0945, 1e-3); // x1
  EXPECT_NEAR(back_grad[6]->_matrix.coeff(0), -0.1673, 1e-3); // x2
  EXPECT_NEAR(back_grad[6]->_matrix.coeff(1), 0.3370, 1e-3);
}
