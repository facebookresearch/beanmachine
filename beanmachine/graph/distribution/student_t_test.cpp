// Copyright (c) Facebook, Inc. and its affiliates.
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
      g.add_distribution(DistributionType::STUDENT_T, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(DistributionType::STUDENT_T, AtomicType::REAL, std::vector<uint>{dof}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(DistributionType::STUDENT_T, AtomicType::REAL, std::vector<uint>{dof, loc}),
      std::invalid_argument);
  // negative test the parents must be a positive, real and a positive
  EXPECT_THROW(
      g.add_distribution(DistributionType::STUDENT_T, AtomicType::REAL, std::vector<uint>{loc, dof, scale}),
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
      DistributionType::STUDENT_T, AtomicType::REAL, std::vector<uint>{dof, loc, scale});
  // test distribution of mean and variance
  auto real_val = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{student_t_dist});
  auto real_sq_val = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means = g.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], LOC, 0.1);
  EXPECT_NEAR(means[1] - means[0]*means[0], (DOF / (DOF - 2)) * SCALE * SCALE, 1.0);
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
  dof = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{
      g.add_distribution(DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  loc = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{
      g.add_distribution(DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{})});
  scale = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{
      g.add_distribution(DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto dof_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{dof, dof});
  auto loc_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{loc, loc});
  auto scale_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{scale, scale});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{
      g.add_distribution(DistributionType::STUDENT_T, AtomicType::REAL,
      std::vector<uint>{dof_sq, loc_sq, scale_sq})});
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
}
