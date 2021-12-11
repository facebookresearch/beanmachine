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

In Bean Machine, samples can be generated from the model via `simulate` (TODO: link to docs). Let's take the earlier model of inferring bias of a coin.
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
                 num_samples=100)
assert isinstance(x_new, MonteCarloSamples)
plot(x_new)
```
*TODO: add plot*


To generate the (empirical) *posterior* predictive distribution $p(x_{new}|x)$, we call `simulate()` on our posterior samples returned by `infer()`.

*TODO: explain what it means to simulate from a posterior represented by samples; is it some sort of re-sampling? Explain how we get the (1, 100, 100) shape below exactly. What is the 1 dimension? Inference chain? Why do we get an inference chain when sampling from samples? Presumably that is not an MCMC method.*

```python
# run inference
posterior = mcmc.infer(queries, obs, num_samples=100)

# generate predictives from our posterior
x_post_pred = simulate(observations.keys(),
                       posterior=posterior,
                       num_samples=100)
assert x_post_pred[queries[0]].shape == (1, 100, 100)
plot(x_post_pred)
```
*TODO: add plot*

Notice that in the posterior predictive case, we pass the observation keys (random variable identifiers) as the "queries" since we are querying about their posterior distribution.

*TODO: the first plot shows both `queries` and `observations.keys()` whereas the second plot only shows `observations`. I am not how the first plot will show this data, but I would assume the point of showing the second plot is to compare it with the first one to see if the posterior on `observations.keys()` looks like the distribution on `observations.keys()` in the first plot. This means that the first plot should show the representation on these variables separately from the distribution on `queries`. In any case, it is not very clear what is being done here so this needs a more clear explanation.*

[^1]: Gelman, A., et al. *Understanding predictive information criteria for Bayesian models*. https://arxiv.org/abs/1307.5928.
