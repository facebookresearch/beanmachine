#include "beanmachine/graph/global/global_state.h"
#include "beanmachine/graph/global/hmc_util.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class GlobalProposer {
 public:
  explicit GlobalProposer() {}
  virtual void warmup(
      double /*acceptance_log_prob*/,
      int /*iteration*/,
      int /*num_warmup_samples*/) {}
  virtual double propose(GlobalState& state, std::mt19937& gen) = 0;
  virtual void initialize(
      GlobalState& /*state*/,
      std::mt19937& /*gen*/,
      int /*num_warmup_samples*/) {}
  virtual ~GlobalProposer() {}
};

class RandomWalkProposer : public GlobalProposer {
 public:
  explicit RandomWalkProposer(double step_size);
  double propose(GlobalState& state, std::mt19937& gen) override;

 private:
  double step_size;
};

class HmcProposer : public GlobalProposer {
 public:
  explicit HmcProposer(
      double path_length,
      double step_size = 0.1,
      double optimal_acceptance_prob = 0.65);
  void initialize(GlobalState& state, std::mt19937& gen, int num_warmup_samples)
      override;
  void warmup(double acceptance_log_prob, int iteration, int num_warmup_samples)
      override;
  double propose(GlobalState& state, std::mt19937& gen) override;

 private:
  StepSizeAdapter step_size_adapter;
  double path_length;
  double step_size;
  double compute_kinetic_energy(Eigen::VectorXd momentum);
  Eigen::VectorXd compute_potential_gradient(GlobalState& state);
};

} // namespace graph
} // namespace beanmachine
