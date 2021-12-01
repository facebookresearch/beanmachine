---
id: compositional_inference
title: 'Block and Compositional Inference'
sidebar_label: 'Block and Compositional Inference'
slug: '/compositional_inference'
---

Single-site inference is not always suitable for models with highly correlated variables because, given a global assignment (or a *state*) in the probability distribution, changing the value of only one variable leads to a new, highly unlikely state that will rarely generate a useful sample. In other words, we may end up at a region of the posterior distribution where individual updates proposing a new value for a single random variable and deciding to accept or reject the new value (based on the Metropolis Hasting rule) are not good enough to be accepted. In these examples, however, if we change the values of a group of random variables together, we may be able to successfully change to another likely state, exploring the posterior more efficiently. Block inference allows Bean Machine to overcome the limitations of single site because highly correlated variables are updated together, allowing for states with higher probabilities.

Referring back to the Gaussian Mixture Model (GMM), we have the following:

```py
@bm.random_variable
def alpha():
    return dist.Dirichlet(torch.ones(K))

@bm.random_variable
def component(i):
    return dist.Categorical(alpha())

@bm.random_variable
def mu(c):
    return dist.MultivariateNormal(
        loc=torch.zeros(2),
        covariance_matrix=10.*torch.eye(2)
   )

@bm.random_variable
def sigma(c):
    return dist.Gamma(1, 1)

@bm.random_variable
def y(i):
    c = component(i)
    return dist.MultivariateNormal(
        loc=mu(c),
        covariance_matrix=sigma(c)**2*torch.eye(2)
   )
```

In the model above, we can either:

  * Use single site inference: where we propose a new value for each random variable, and accept/reject them individually using the Metropolis Hastings rule.
  * Use block inference: where we block random variables together, sequentially propose a new value for the random variables in the block and accept/reject all proposed values together. For instance, if the proposed value of component(i), which is the component assignment for the ith data point, is to go from c to c', then y(i) is no longer a child of mu(c) and sigma(c) and is instead a child of mu(c') and sigma(c'). The likelihood of the world with the component(i)‘s new proposal alone is low, because, all mu(c), sigma(c), mu(c') and sigma(c') are all sampled with the assumption that y(i) was observed from component, c. Our solution here is to propose new values for mu(c), sigma(c), mu(c'), sigma(c') and component(i) and accept/reject all 5 values together.

To run block inference, you can:

```py
mh = bm.CompositionalInference()
mh.add_sequential_proposer([component, sigma, mu])

samples = mh.infer(queries, observations, n_samples, num_chains)
```

Note that the user does not need to tell Bean Machine which `mu` and `sigma` need to be grouped with the component. Bean Machine only requires the random variable families to be passed to `add_sequential_proposer`. The Bean Machine inference engine can then use the model dependency structure after re-sampling `component(i)` from $c$ to $c'$ to also re-sample all `mu` and `sigma` in old `component(i)`‘s Markov Blanket, `mu($c$)` and `sigma($c$)` and `mu` and `sigma` in the new Markov Blanket, `mu($c'$)` and `sigma($c'$)`.
