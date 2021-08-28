#include <memory>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

/*
 * An adapter to go from a base proposer producing a probability p
 * to a new proposer that produces a Dirichlet sample (p, 1 - p).
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
