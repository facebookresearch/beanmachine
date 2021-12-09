/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/bimixture.h"
#include "beanmachine/graph/util.h"

/*
A MACRO function that impletments the generic logic of sampling from
a bi-mixture distribution.
:param dtype_sampler: A method that takes a std::mt19937
                      reference and returns a dtype sample.
*/
#define GENERIC_DTYPE_SAMPLER(dtype_sampler)                                \
  auto bern_dist = std::bernoulli_distribution(in_nodes[0]->value._double); \
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);    \
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);    \
  if (bern_dist(gen)) {                                                     \
    return d1->dtype_sampler(gen);                                          \
  } else {                                                                  \
    return d2->dtype_sampler(gen);                                          \
  }

// A MACRO that prepares the intermediate values for gradient propagation
#define BIMIX_PREPARE_GRAD()                                             \
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]); \
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]); \
  double p = in_nodes[0]->value._double;                                 \
  double logf1 = d1->log_prob(value), logf2 = d2->log_prob(value);       \
  double max_logfi = std::max(logf1, logf2);                             \
  double f1 = std::exp(logf1 - max_logfi);                               \
  double f2 = std::exp(logf2 - max_logfi);                               \
  double f = p * f1 + (1.0 - p) * f2;

#define BIMIX_PREPARE_GRAD_IID()                                             \
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX); \
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);     \
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);     \
  double p = in_nodes[0]->value._double;                                     \
  Eigen::MatrixXd logf1;                                                     \
  Eigen::MatrixXd logf2;                                                     \
  d1->log_prob_iid(value, logf1);                                            \
  d2->log_prob_iid(value, logf2);                                            \
  Eigen::MatrixXd max_logfi = logf1.cwiseMax(logf2);                         \
  Eigen::MatrixXd f1 = (logf1.array() - max_logfi.array()).exp();            \
  Eigen::MatrixXd f2 = (logf2.array() - max_logfi.array()).exp();            \
  Eigen::MatrixXd f = p * f1 + (1.0 - p) * f2;

namespace beanmachine {
namespace distribution {

using namespace graph;

Bimixture::Bimixture(ValueType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::BIMIXTURE, sample_type) {
  // a Bimixture distribution has three parents:
  // [the probability of sampling from Dist_1, node for Dist_1, node for Dist_2]
  if (in_nodes.size() != 3) {
    throw std::invalid_argument(
        "Bimixture distribution must have exactly three parents");
  }
  if (in_nodes[0]->value.type != AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "the first parent for bimixture distribution must be a probability");
  }
  if (in_nodes[1]->node_type != NodeType::DISTRIBUTION or
      in_nodes[2]->node_type != NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
        "the 2nd and 3rd parent for bimixture distribution must be distributions");
  }
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);
  if (sample_type != d1->sample_type or sample_type != d2->sample_type) {
    throw std::invalid_argument(
        "sample type must be consistent with the distribution parents");
  }
}

Bimixture::Bimixture(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Bimixture(ValueType(sample_type), in_nodes) {}

bool Bimixture::_bool_sampler(std::mt19937& gen) const {
  GENERIC_DTYPE_SAMPLER(_bool_sampler)
}

double Bimixture::_double_sampler(std::mt19937& gen) const {
    GENERIC_DTYPE_SAMPLER(_double_sampler)}

natural_t Bimixture::_natural_sampler(std::mt19937& gen) const {
  GENERIC_DTYPE_SAMPLER(_natural_sampler)
}

// log_prob(x | p, f1, f2) i.e. log(f) = logsumexp(log(p) + log(f1), log(1-p) +
// log(f2))
double Bimixture::log_prob(const graph::NodeValue& value) const {
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    double z1 = std::log(in_nodes[0]->value._double) + d1->log_prob(value);
    double z2 =
        std::log(1.0 - in_nodes[0]->value._double) + d2->log_prob(value);
    return util::log_sum_exp(std::vector<double>{z1, z2});
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    Eigen::MatrixXd log_probs;
    log_prob_iid(value, log_probs);
    return log_probs.sum();
  } else {
    throw std::runtime_error(
        "Bimixture::log_prob applied to invalid variable type");
  }
}

void Bimixture::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double p = in_nodes[0]->value._double;
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);
  Eigen::MatrixXd logf1;
  Eigen::MatrixXd logf2;
  d1->log_prob_iid(value, logf1);
  d2->log_prob_iid(value, logf2);
  logf1.array() += std::log(p);
  logf2.array() += std::log(1.0 - p);
  log_probs = logf1.binaryExpr(logf2, util::BinaryLogSumExp());
}

// f' = p * f1' + (1-p) * f2' = p * f1 * log(f1)' + (1-p) * f2 * log(f2)'
// f'' = p * f1'' + (1-p) * f2''
//     = sum_i={1,2} pi * (fi' * log(fi)' + fi * log(fi)''), let p1 = p, p2 = 1
//     - p = sum_i={1,2} pi * (fi * (log(fi)')^2 + fi * log(fi)''),
// grad1 w.r.t. x: log(f)' = f'/f
// grad2 w.r.t. x: log(f)'' = f''/f - (f'/f)^2 = f''/f - (log(f)')^2
void Bimixture::gradient_log_prob_value(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  BIMIX_PREPARE_GRAD()
  double logf1_grad1 = 0.0, logf1_grad2 = 0.0, logf2_grad1 = 0.0,
         logf2_grad2 = 0.0;
  d1->gradient_log_prob_value(value, logf1_grad1, logf1_grad2);
  d2->gradient_log_prob_value(value, logf2_grad1, logf2_grad2);

  double logf_grad1 = (p * f1 * logf1_grad1 + (1 - p) * f2 * logf2_grad1) / f;
  double f_grad2 = p * f1 * (logf1_grad1 * logf1_grad1 + logf1_grad2) +
      (1 - p) * f2 * (logf2_grad1 * logf2_grad1 + logf2_grad2);

  grad1 += logf_grad1;
  grad2 += f_grad2 / f - logf_grad1 * logf_grad1;
}

// let q1 be log(f1), q2 be log(f2)
// source node: z, G(z) = (p, q1, q2), F(p, q1, q2) = log(p * exp(q1) + (1-p) *
// exp(q2)) grad1 w.r.t. params: dF/dz = Jacob_F @ Jacob_G grad2 w.r.t. params:
// d^2F/dz^2 = Jacob_G^T @ Hess_F @ Jacob_G + Jacob_F @ d^2G/dz^2
// where Jacob_F = ((f1 - f2) / f, p * f1 / f, (1-p) * f2 / f)
// Hess_f =            - (f1 - f2)^2 / f^2,
//       f1 / f - p * f1 * (f1 - f2) / f^2, p * f1 / f - (p * f1 / f)^2,
// - f2 / f - (1-p) * f2 * (f1 - f2) / f^2, - p * f1 * (1-p) * f2 / f^2, (1-p) *
// f2 / f - ((1-p) * f2 / f)^2
void Bimixture::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  BIMIX_PREPARE_GRAD()
  Eigen::Matrix<double, 1, 3> Jacob_F;
  double Jf0 = (f1 - f2) / f, Jf1 = p * f1 / f, Jf2 = (1 - p) * f2 / f;
  Jacob_F << Jf0, Jf1, Jf2;
  Eigen::Matrix<double, 3, 3> Hess_F;
  // only need to compute the upper triangle, Hess_F is symmetric
  Hess_F << -Jf0 * Jf0, f1 / f - Jf0 * Jf1, -f2 / f - Jf0 * Jf2, 0.0,
      Jf1 - Jf1 * Jf1, -Jf1 * Jf2, 0.0, 0.0, Jf2 - Jf2 * Jf2;
  double Jg0 = in_nodes[0]->grad1, Jg1 = 0.0, Jg2 = 0.0;
  double J2g0 = in_nodes[0]->grad2, J2g1 = 0.0, J2g2 = 0.0;
  d1->gradient_log_prob_param(value, Jg1, J2g1);
  d2->gradient_log_prob_param(value, Jg2, J2g2);
  Eigen::Matrix<double, 1, 3> Jacob_G;
  Jacob_G << Jg0, Jg1, Jg2;
  Eigen::Matrix<double, 1, 3> Grad2_G;
  Grad2_G << J2g0, J2g1, J2g2;

  grad1 += (Jacob_F * Jacob_G.transpose()).coeff(0, 0);
  grad2 +=
      (Jacob_G * Hess_F.selfadjointView<Eigen::Upper>() * Jacob_G.transpose() +
       Jacob_F * Grad2_G.transpose())
          .coeff(0, 0);
}

// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]

// for x being the value:
// dlog(f)/dx = (df/dx)/f
//  = (p * df1/dx + (1-p) * df2/dx ) / f
//  = p * f1/f * log(f1)' + (1-p) * f2/f * log(f2)'
void Bimixture::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  BIMIX_PREPARE_GRAD()
  d1->backward_value(value, back_grad, adjunct * p * f1 / f);
  d2->backward_value(value, back_grad, adjunct * (1 - p) * f2 / f);
}

void Bimixture::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  BIMIX_PREPARE_GRAD_IID()
  f1 = p * f1.array() / f.array();
  f2 = (1 - p) * f2.array() / f.array();
  d1->backward_value_iid(value, back_grad, f1);
  d2->backward_value_iid(value, back_grad, f2);
}

void Bimixture::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  BIMIX_PREPARE_GRAD_IID()
  f1 = p * f1.array() / f.array() * adjunct.array();
  f2 = (1 - p) * f2.array() / f.array() * adjunct.array();
  d1->backward_value_iid(value, back_grad, f1);
  d2->backward_value_iid(value, back_grad, f2);
}

// for x being the parameter:
// backprop thru p: dlog(f)/dx = dlog(f)/dp * dp/dx
//   = (f1 - f2) / f * dp/dx
// backprop thru fi, i in {1, 2}: dlog(f)/dx = dlog(f)/dlog(fi) * dlog(fi)/dx
// dlog(f)/dlog(f1) = p * f1 / f
// dlog(f)/dlog(f2) = p * (1 - p) * f2 / f
void Bimixture::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  BIMIX_PREPARE_GRAD()
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double += adjunct * (f1 - f2) / f;
  }
  d1->backward_param(value, adjunct * p * f1 / f);
  d2->backward_param(value, adjunct * (1 - p) * f2 / f);
}

void Bimixture::backward_param_iid(const graph::NodeValue& value) const {
  BIMIX_PREPARE_GRAD_IID()
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double +=
        ((f1.array() - f2.array()) / f.array()).sum();
  }
  f1 = p * f1.array() / f.array();
  f2 = (1 - p) * f2.array() / f.array();
  d1->backward_param_iid(value, f1);
  d2->backward_param_iid(value, f2);
}

void Bimixture::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  BIMIX_PREPARE_GRAD_IID()
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double +=
        ((f1.array() - f2.array()) / f.array() * adjunct.array()).sum();
  }
  f1 = p * f1.array() / f.array() * adjunct.array();
  f2 = (1 - p) * f2.array() / f.array() * adjunct.array();
  d1->backward_param_iid(value, f1);
  d2->backward_param_iid(value, f2);
}

} // namespace distribution
} // namespace beanmachine
