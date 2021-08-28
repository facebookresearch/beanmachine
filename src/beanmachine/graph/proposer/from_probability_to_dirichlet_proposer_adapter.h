#include <memory>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

/*
 * A proposer using a base proposer that produces a probability p
 * to produce a Dirichlet sample (p, 1 - p).
 */
class FromProbabilityToDirichletProposerAdapter : public Proposer {
 public:
  explicit FromProbabilityToDirichletProposerAdapter(
      std::unique_ptr<Proposer> probability_proposer)
      : probability_proposer(std::move(probability_proposer)) {}

  virtual graph::NodeValue sample(std::mt19937& gen) const override;

  virtual double log_prob(graph::NodeValue& value) const override;

 private:
  std::unique_ptr<Proposer> probability_proposer;
};

} // namespace proposer
} // namespace beanmachine
