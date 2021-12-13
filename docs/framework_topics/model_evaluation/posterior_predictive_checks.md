---
id: posterior_predictive_checks
title: 'Posterior Predictive Checks'
sidebar_label: 'Posterior Predictive Checks'
slug: '/posterior_predictive_checks'
---

Evaluating probabilistic models is a challenging effort and an open research problem.  A common way to evaluate how well your model fits the data is to do *posterior predictive checking* [^1], i.e. how well a model's predictions match the data.

$$
p(x_{new} | x) = E_{z\sim p(z|x)} p(x_{new}|z)
$$

In Bean Machine, samples can be generated from the model via [`simulate`](https://beanmachine.org/api/beanmachine.ppl.inference.predictive.html?beanmachine.ppl.inference.predictive.Predictive.simulate#beanmachine.ppl.inference.predictive.simulate). Let's take the earlier model of inferring bias of a coin.
```python
@bm.random_variable
def coin_bias():
   return dist.Beta(2, 2)

@bm.random_variable
def coin_toss(i):
    return dist.Bernoulli(coin_bias())
```
We can generate synthetic data from our model as follows:
```python
true_coin_bias = 0.75
coin_true_distribution = dist.Bernoulli(true_coin_bias)
flip_count = 100

queries = [coin_bias()]
observations = {coin_toss(i): coin_data[i] for i in range(flip_count)}

# simulate from the prior
x_new = simulate(queries + list(observations.keys()),
                 num_samples=1)
assert isinstance(x_new, MonteCarloSamples)

# Print out some generated coin tosses from our model.
print([x_new[coin_toss(i)] for i in range(flip_count)])
```

This is also known as a prior predictive. Now lets say we've run inference to learn more sensible distributions for our random variables, and we'd like to generate new data with our learned model.
To generate the (empirical) *posterior* predictive distribution $p(x_{new}|x)$, we call `simulate()` on our posterior samples returned by `infer()`. Note that this time, the first argument
to `simulate` is only the observations since those are the values we are querying. The samples for the other random variables have already been collected from inference.

```python
# run inference
num_infer_samples = 30
num_sim_samples = 50
posterior = mcmc.infer(queries, obs, num_samples=30)

# generate predictives from our posterior
x_post_pred = simulate(observations.keys(),
                       posterior=posterior,
                       num_samples=100)
assert x_post_pred[queries[0]].shape == (1, num_infer_samples, num_sim_samples)
```

Note the shape here; since `simulate` is an inference subroutine under the hood (one in which we just forward sample the model), in theory, it can be run with multiple chains. For this example,
we only have one chain. Then for each Monte Carlo sample of our posterior, we sample `num_sim_samples` many coin flips. We can then use [`Empirical`](https://beanmachine.org/api/beanmachine.ppl.html#beanmachine.ppl.empirical)
to sample from this resulting bag of samples. From here, one can compute various statistics on the posterior predictive data and compare with the ground truth data to assess model fitness.

[^1]: Gelman, A., et al. *Understanding predictive information criteria for Bayesian models*. https://arxiv.org/abs/1307.5928.
