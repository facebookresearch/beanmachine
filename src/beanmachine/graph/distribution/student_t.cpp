// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/student_t.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

StudentT::StudentT(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::STUDENT_T, sample_type) {
  // a StudentT distribution has three parents
  // n (degrees of freedom) > 0 ; l (location) -> real; scale -> positive real
  if (in_nodes.size() != 3) {
    throw std::invalid_argument(
        "StudentT distribution must have exactly three parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL or
      in_nodes[1]->value.type != graph::AtomicType::REAL or
      in_nodes[2]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "StudentT parents must have parents (positive, real, positive)");
  }
  // only real-valued samples are possible
  if (sample_type != AtomicType::REAL) {
    throw std::invalid_argument(
        "StudentT distribution produces real number samples");
  }
}

double StudentT::_double_sampler(std::mt19937& gen) const {
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  std::student_t_distribution<double> dist(n);
  return l + dist(gen) * s;
}

// log_prob of a student-t with parameters n, l, s
// (degrees of freedom, location, and scale respectively):
// f(x, n, l, s) =
//     log(G((n+1)/2)) - log(G(n/2)) - 0.5 log(n) - 0.5 log(pi) - log(s)
//     -(n+1)/2 [ log(n s^2 + (x-l)^2) - log(n) - 2 log(s) ]
//
// df / dx = = -(n+1) (x-l) / (n s^2 + (x-l)^2)
// d2f/dx2 = -(n+1) [ 1/(n s^2 + (x-l)^2)   -  2(x-l)^2 / (n s^2 + (x-l)^2)^2 ]
//
// df / dn = 0.5 digamma((n+1)/2) - 0.5 digamma(n/2) - 0.5/n
//           -0.5 [ log(n s^2 + (x-l)^2) - log(n) - 2 log(s) ]
//           -0.5 (n+1) [s^2/(n s^2 + (x-l)^2) - 1/n ]
// d2f/dn2 = 0.25 polygamma(1, (n+1)/2) - 0.25 * polygamma(1, n/2) + 0.5/n^2
//           -0.5 [ s^2 / (n s^2 + (x-l)^2) - 1/n ]
//           -0.5 [ s^2 / (n s^2 + (x-l)^2) - 1/n ]
//           -0.5 (n+1) [ -s^4 / (n s^2 + (x-l)^2)^2 + 1/n^2 ]
// df / dl   = (n+1) (x-l) / (n s^2 + (x-l)^2)
// d2f / dl2 = - (n+1) / (n s^2 + (x-l)^2) + 2 (n+1) (x-l)^2 / (n s^2 +
// (x-l)^2)^2 df / ds  = -1/s -(n+1) ( n s / (n s^2 + (x-l)^2)  - 1/s ) d2f ds2
// = 1/s^2 -(n+1)( n / (n s^2 + (x-l)^2) - 2 n^2 s^2 / (n s^2 + (x-l)^2)^2  +
// 1/s^2)

double StudentT::log_prob(const NodeValue& value) const {
  double x = value._double;
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  return std::lgamma((n + 1) / 2) - std::lgamma(n / 2) - 0.5 * std::log(n) -
      0.5 * std::log(M_PI) - std::log(s) -
      ((n + 1) / 2) *
      (std::log(n * s * s + (x - l) * (x - l)) - std::log(n) - 2 * std::log(s));
}

void StudentT::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  double x = value._double;
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  double n_s_sq_p_x_m_l_sq = n * s * s + (x - l) * (x - l); // n s^2 + (x-l)^2
  grad1 += -(n + 1) * (x - l) / n_s_sq_p_x_m_l_sq;
  grad2 += -(n + 1) *
      (1 / n_s_sq_p_x_m_l_sq -
       2 * (x - l) * (x - l) / (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq));
}

void StudentT::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  const double x = value._double;
  const double n = in_nodes[0]->value._double;
  const double l = in_nodes[1]->value._double;
  const double s = in_nodes[2]->value._double;
  const double n_s_sq_p_x_m_l_sq =
      n * s * s + (x - l) * (x - l); // n s^2 + (x-l)^2
  // We will compute the gradients w.r.t. each of the parameters only if
  // the gradients of the parameters w.r.t. the source index is non-zero
  double n_grad = in_nodes[0]->grad1;
  double n_grad2 = in_nodes[0]->grad2;
  if (n_grad != 0 or n_grad2 != 0) {
    double grad_n = 0.5 * util::polygamma(0, (n + 1) / 2) -
        0.5 * util::polygamma(0, n / 2) - 0.5 / n -
        0.5 * (std::log(n_s_sq_p_x_m_l_sq) - std::log(n) - 2 * std::log(s)) -
        0.5 * (n + 1) * (s * s / n_s_sq_p_x_m_l_sq - 1 / n);
    double grad2_n2 = 0.25 * util::polygamma(1, (n + 1) / 2) -
        0.25 * util::polygamma(1, n / 2) + 0.5 / (n * n) -
        (s * s / n_s_sq_p_x_m_l_sq - 1 / n) -
        0.5 * (n + 1) *
            (-std::pow(s, 4) / (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq) +
             1 / (n * n));
    grad1 += grad_n * n_grad;
    grad2 += grad2_n2 * n_grad * n_grad + grad_n * n_grad2;
  }
  double l_grad = in_nodes[1]->grad1;
  double l_grad2 = in_nodes[1]->grad2;
  if (l_grad != 0 or l_grad2 != 0) {
    double grad_l = (n + 1) * (x - l) / n_s_sq_p_x_m_l_sq;
    double grad2_l2 = -(n + 1) / n_s_sq_p_x_m_l_sq +
        2 * (n + 1) * (x - l) * (x - l) /
            (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq);
    grad1 += grad_l * l_grad;
    grad2 += grad2_l2 * l_grad * l_grad + grad_l * l_grad2;
  }
  double s_grad = in_nodes[2]->grad1;
  double s_grad2 = in_nodes[2]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double grad_s = -1 / s - (n + 1) * (n * s / n_s_sq_p_x_m_l_sq - 1 / s);
    double grad2_s2 = 1 / (s * s) -
        (n + 1) *
            (n / n_s_sq_p_x_m_l_sq -
             2 * n * n * s * s / (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq) +
             1 / (s * s));
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

} // namespace distribution
} // namespace beanmachine
