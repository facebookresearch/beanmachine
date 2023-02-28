import beanmachine.ppl as bm
from beanmachine.ppl.experimental.mixedhmc import inference as mix
import torch.distributions as dist
import torch


class CategoricalMixtureModel:
    @bm.random_variable
    def c():
        probs = torch.tensor([0.15, 0.3, 0.3, 0.25])
        return dist.Categorical(probs)

    @bm.random_variable
    def x():
        locs = torch.tensor([-2., 0., 2., 4.])
        return dist.Normal(locs[c()], 0.5)


def test_margins():
    model = CategoricalMixtureModel()

    mix_samples = mix.MixedHMC(max_step_size=1.0, num_discrete_updates=1, trajectory_length=1.2).infer(
        queries=[model.x()],
        observations={},
        num_adaptive_samples=1000,
        num_samples=4000,
    )

    comp_samples = bm.CompositionalInference(
        {model.x: bm.SingleSiteUniformMetropolisHastings()}
    ).infer(
        queries=[model.x()],
        observations={},
        num_adaptive_samples=1000,
        num_samples=4000,
    )

    assert comp_samples[model.x()].shape == mix_samples[model.x()].shape
