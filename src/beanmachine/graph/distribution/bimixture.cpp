// Copyright (c) Facebook, Inc. and its affiliates.
#include <string>
#include <random>
#include <cmath>

#include "beanmachine/graph/distribution/bimixture.h"
#include "beanmachine/graph/util.h"

/*
A MACRO function that impletments the generic logic of sampling from
a bi-mixture distribution.
:param dtype_sampler: A method that takes a std::mt19937
                      reference and returns a dtype sample.
*/
#define GENERIC_DTYPE_SAMPLER(dtype_sampler) \
  auto bern_dist = std::bernoulli_distribution(in_nodes[0]->value._double); \
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);    \
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);    \
  if (bern_dist(gen)) {                                                     \
    return d1->dtype_sampler(gen);                                          \
  } else {                                                                  \
    return d2->dtype_sampler(gen);                                          \
  }                                                                         \

namespace beanmachine {
namespace distribution {

using namespace graph;

Bimixture::Bimixture(
    ValueType sample_type,
    const std::vector<Node*>& in_nodes)
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

Bimixture::Bimixture(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Bimixture(ValueType(sample_type), in_nodes) {
}

bool Bimixture::_bool_sampler(std::mt19937& gen) const {
  GENERIC_DTYPE_SAMPLER(_bool_sampler)
}

double Bimixture::_double_sampler(std::mt19937& gen) const {
  GENERIC_DTYPE_SAMPLER(_double_sampler)
}

natural_t Bimixture::_natural_sampler(std::mt19937& gen) const {
  GENERIC_DTYPE_SAMPLER(_natural_sampler)
}

// log_prob(x | p, f1, f2) i.e. log(f) = logsumexp(log(p) + log(f1), log(1-p) + log(f2))
double Bimixture::log_prob(const graph::AtomicValue& value) const {
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);
  double z1 = std::log(in_nodes[0]->value._double) + d1->log_prob(value);
  double z2 = std::log(1.0 - in_nodes[0]->value._double) + d2->log_prob(value);
  return util::log_sum_exp(std::vector<double>{z1, z2});
}

// f' = p * f1' + (1-p) * f2' = p * f1 * log(f1)' + (1-p) * f2 * log(f2)'
// f'' = p * f1'' + (1-p) * f2''
//     = sum_i={1,2} pi * (fi' * log(fi)' + fi * log(fi)''), let p1 = p, p2 = 1 - p
//     = sum_i={1,2} pi * (fi * (log(fi)')^2 + fi * log(fi)''),
// grad1 w.r.t. x: log(f)' = f'/f
// grad2 w.r.t. x: log(f)'' = f''/f - (f'/f)^2 = f''/f - (log(f)')^2
void Bimixture::gradient_log_prob_value(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);
  double p = in_nodes[0]->value._double;
  double logf1 = d1->log_prob(value), logf2 = d2->log_prob(value);
  double max_logfi = std::max(logf1, logf2);
  // scale f, f1, f2 by 1/max(f1, f2) for numerical stability
  double f1 = std::exp(logf1 - max_logfi);
  double f2 = std::exp(logf2 - max_logfi);
  double f = p * f1 + (1.0 - p) * f2;
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
// source node: z, G(z) = (p, q1, q2), F(p, q1, q2) = log(p * exp(q1) + (1-p) * exp(q2))
// grad1 w.r.t. params:
// dF/dz = Jacob_F @ Jacob_G
// grad2 w.r.t. params:
// d^2F/dz^2 = Jacob_G^T @ Hess_F @ Jacob_G + Jacob_F @ d^2G/dz^2
// where Jacob_F = ((f1 - f2) / f, p * f1 / f, (1-p) * f2 / f)
// Hess_f =            - (f1 - f2)^2 / f^2,
//       f1 / f - p * f1 * (f1 - f2) / f^2, p * f1 / f - (p * f1 / f)^2,
// - f2 / f - (1-p) * f2 * (f1 - f2) / f^2, - p * f1 * (1-p) * f2 / f^2, (1-p) * f2 / f - ((1-p) * f2 / f)^2
void Bimixture::gradient_log_prob_param(
    const graph::AtomicValue& value, double& grad1, double& grad2) const {
  auto d1 = static_cast<const distribution::Distribution*>(in_nodes[1]);
  auto d2 = static_cast<const distribution::Distribution*>(in_nodes[2]);
  double p = in_nodes[0]->value._double;
  double logf1 = d1->log_prob(value), logf2 = d2->log_prob(value);
  double max_logfi = std::max(logf1, logf2);
  // scale f, f1, f2 by 1/max(f1, f2) for numerical stability
  double f1 = std::exp(logf1 - max_logfi);
  double f2 = std::exp(logf2 - max_logfi);
  double f = p * f1 + (1.0 - p) * f2;
  Eigen::Matrix<double, 1, 3> Jacob_F;
  double Jf0 = (f1 - f2) / f, Jf1 = p * f1 / f, Jf2 = (1-p) * f2 / f;
  Jacob_F << Jf0, Jf1, Jf2;
  Eigen::Matrix<double, 3, 3> Hess_F;
  // only need to compute the upper triangle, Hess_F is symmetric
  Hess_F << - Jf0 * Jf0, f1 / f - Jf0 * Jf1, - f2/f - Jf0 * Jf2,
                    0.0,    Jf1 - Jf1 * Jf1,        - Jf1 * Jf2,
                    0.0,                0.0,    Jf2 - Jf2 * Jf2;
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
       Jacob_F * Grad2_G.transpose()).coeff(0, 0);
}

} // namespace distribution
} // namespace beanmachine
