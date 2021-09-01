#include "beanmachine/graph/global/global_mh.h"

namespace beanmachine {
namespace graph {

HMC::HMC(Graph& g, uint seed, double path_length, double step_size)
    : GlobalMH(g, seed) {
  proposer = std::make_unique<HmcProposer>(HmcProposer(path_length, step_size));
}

HmcProposer::HmcProposer(double path_length, double step_size)
    : GlobalProposer() {
  this->path_length = path_length;
  this->step_size = step_size;
}

double HmcProposer::compute_kinetic_energy(Eigen::VectorXd p) {
  return 0.5 * (p * p).sum();
}

Eigen::VectorXd HmcProposer::compute_potential_gradient(GlobalState& state) {
  state.update_backgrad();
  Eigen::VectorXd grad1;
  state.get_flattened_unconstrained_grads(grad1);
  return -grad1;
}

double HmcProposer::propose(GlobalState& state, std::mt19937& gen) {
  Eigen::VectorXd q;
  state.get_flattened_unconstrained_values(q);
  state.update_log_prob();
  double initial_U = -state.get_log_prob();

  Eigen::VectorXd p(q.size());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < p.size(); i++) {
    p[i] = dist(gen);
  }
  double initial_K = compute_kinetic_energy(p);

  int num_steps = ceil(path_length / step_size);

  // momentum half-step
  Eigen::VectorXd grad_U = compute_potential_gradient(state);
  p = p - step_size * grad_U / 2;
  for (int i = 0; i < num_steps; i++) {
    // position full-step
    q = q + step_size * p;

    // momentum step
    state.set_flattened_unconstrained_values(q);
    grad_U = compute_potential_gradient(state);
    if (i < num_steps - 1) {
      // full-step
      p = p - step_size * grad_U;
    } else {
      // half-step at the last iteration
      p = p - step_size * grad_U / 2;
    }
  }

  double final_K = compute_kinetic_energy(p);
  state.update_log_prob();
  double final_U = -state.get_log_prob();
  return initial_U - final_U + initial_K - final_K;
}

} // namespace graph
} // namespace beanmachine
