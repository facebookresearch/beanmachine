import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.variable import Variable


def test_log_prob():
    var1 = Variable(
        transformed_value=torch.zeros(3),
        transform=dist.identity_transform,
        distribution=dist.Bernoulli(0.8),
    )
    # verify that the cached property `log_prob` is recomputed when we replace the
    # fields of a Variable
    var2 = var1.replace(transformed_value=torch.ones(3))
    assert var1.log_prob.sum() < var2.log_prob.sum()

    var3 = var1.replace(distribution=dist.Normal(0.0, 1.0))
    assert var1.log_prob.sum() < var3.log_prob.sum()

    var4 = var1.replace(distribution=dist.Categorical(logits=torch.rand(2, 4)))
    assert torch.all(torch.isinf(var4.log_prob))
