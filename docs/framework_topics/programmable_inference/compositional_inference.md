# Block inference

Single-site inference is not always suitable for models with highly correlated variables as no single variable can be accepted individually. During inference, we may end up at part of the posterior, where individual *Metropolis Hastings updates*, proposing a new value for a single random variable and deciding to accept/reject the new value based on the Metropolis Hasting rule, are not good enough to be accepted. In these examples, however, if we move a group of random variables together, we may be able to successfully move and explore the posterior more efficiently. Block inference allows Bean Machine to overcome the limitations of single site because highly correlated variables are updated together, allowing for worlds with higher probabilities.

Referring back to the Gaussian Mixture Model (GMM), we have the following:

```py
@bm.random_variable
def alpha():
    return dist.Dirichlet(torch.ones(K))

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
def component(i):
    return dist.Categorical(alpha())

@bm.random_variable
def y(i):
    c = component(i).item()
    return dist.MultivariateNormal(
        loc=mu(c),
        covariance_matrix=sigma(c)**2*torch.eye(2)
   )
```

In the model above, we can either:

  * Use single site inference: where we propose a new value for each random variable, and accept/reject them individually using the Metropolis Hasting rule.
  * Use block inference: where we block random variables together, sequentially propose a new value for the random variables in the block and accept/reject all proposed values together. For instance, if the proposed value of component(i), which is the component assignment for the ith data point, is to go from c to c’, then y(i) is no longer a child of mu(c) and sigma(c) and is instead a child of mu(c’) and sigma(c’). The likelihood of the world with the component(i)‘s new proposal alone is low, because, all mu(c), sigma(c), mu(c’) and sigma(c’) are all sampled with the assumption that y(i) was observed from component, c. Our solution here is to propose new values for mu(c), sigma(c), mu(c’), sigma(c’) and component(i) and accept/reject all 5 values together.

To run inference with Block, you can:

```py
mh = bm.CompositionalInference()
mh.add_sequential_proposer([component, sigma, mu])

samples = mh.infer(queries, observations, n_samples, num_chains)
```

Note that, the user does not need to tell Bean Machine which mu and sigma needs to be grouped with the component. Bean Machine only requires the random variable families to be passed to add_sequential_proposer. Bean Machine inference engine can then use the model dependency structure after re-sampling component(i) from c to c’ to also re-sample all mu’s and sigma’s in old component(i)‘s Markov Blanket, mu(c) and sigma(c) and mu’s and sigma’s in the new Markov Blanket, mu(c’) and sigma(c’).
