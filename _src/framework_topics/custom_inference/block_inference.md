---
id: block_inference
title: 'Block Inference'
sidebar_label: 'Block Inference'
slug: '/block_inference'
---

Single-site inference in Bean Machine is a powerful abstraction that allows the inference engine to separately sample values for random variables in your model. While efficient in sampling high-dimensional models, single-site inference may not be suitable for models with highly correlated random variables. This is where Bean Machine's `CompositionalInference` API becomes handy: it allows us to "block" multiple nodes together and make proposals for them jointly.

## Motivation

To understand the issue better, let's walk through an example. Let's say we have two random variables $X$, $Y$ whose values are $x$ and $y$, and we'd like to move these values to $x'$ and $y'$. Using single site inference, we can move from $(x, y)$ to $(x', y')$ with either of these series of updates:

1. $(x, y) \to (x', y) \to (x', y')$
2. $(x, y) \to (x, y') \to (x', y')$

If $X$ and $Y$ are strongly correlated, e.g., if both $p(x, y)$ and $p(x', y')$ are high, but the intermediate stage $p(x', y)$ and $p(x, y')$ are low, then the single site inference methods can be stuck in $(x, y)$, because the acceptance probability for transitioning out of the initial state is going to be very low for either of these two paths. This can lead to under-exploration of $(x', y')$, which will not happen if we block $X$ and $Y$ together, which can move from $(x, y) \to (x', y')$ in a single Metropolis-Hastings step.


:::tip

If you haven't already read the docs on [Compositional Inference](compositional_inference.md), please read those first.

:::

## Configuring Block Inference

To understand how to run block inference in Bean Machine, let's consider the discrete Hidden Markov Model (discrete HMM) example below:

```py
# alpha, beta, rho, nu, and init are externally-defined constants.

@bm.random_variable
def mu(k):
    return Normal(alpha, beta)

@bm.random_variable
def sigma(k):
    return Gamma(nu, rho)

@bm.random_variable
def theta(k):
    return Dirichlet(kappa)

@bm.random_variable
def x(i):
    return Categorical(theta(x(i - 1)) if i else init)

@bm.random_variable
def y(i):
    return Normal(mu(x(i)), sigma(x(i)))
```

This HMM describes a process with categorical latent states `x`, transition probabilities `theta`, and observed states `y` with emission probabilities determined by `mu` and `sigma`. Depending on the value of `theta`, the hidden state at each time step, `x(i)`, can be hightly correlated with the hidden state at the previous time step, `x(i-1)`. Therefore, we might want to block all instances of $x$ into a single block and propose new values for them jointly. Let's say we also want the parameters for emission probabilities, `mu` and `sigma`, to be updated jointly as well.

### Defining a block

You may recall that with [Compositional Inference](compositional_inference.md#configuring-your-own-inference), we can mix-and-match inference methods by providing a mapping from random variable families to inference algorithms through the `inference_dict` argument. The syntax for "blocking" multiple nodes together is similar: instead of having a single random variable as a key, you can pass a tuple of random variable families instead. For example:

```py
bm.CompositionalInference({
    (x,): bm.SingleSiteAncestralMetropolisHastings(),
    (mu, sigma): bm.GlobalNoUTurnSampler(),
})
```

The code snippet above is going to create two blocks: one for all instances of `x`, which will be inferred with `SingleSiteAncestralMetropolisHastings()`, and another for all instances of `mu` and `sigma`, which will be inferred with `GlobalNoUTurnSampler()`. Random variable families that are not specified in the dictionary will fall back to the [default inference methods](compositional_inference.md#default-inference-methods) and run without blocking.

Note that even though single site inference algorithms only update one node at a time, they can still be used to update a block of nodes. What is going to happen internally is that instead of accepting or rejecting a single site proposal immediately after it is made, we condition on it to compute the next proposal, and repeat this process for the remaining nodes in a block. After we are done with all nodes in a block, we then compute the Metropolis-Hastings acceptance probability as if the proposals are made in a single step. (Be aware that many of the single site algorithm only works well when the number of nodes in a block is low, as they might have trouble updating the samples as dimension increases.) On the other hand, multi site inference algorithms, such as `GlobalNoUTurnSampler` that we are using here, can make proposal for a set of nodes in one go and can take advantage of correlation between multiple nodes. To learn more about the distinctions between the two types of inference methods and how to define your own algorithms, see [Custom Proposers](custom_proposers.md).

### Mixing multiple inference methods in a block

Sometimes you might want to use different algorithms to update different random variable families, but stil have them group together in a single block. To do so, you can pass a tuple of inference methods as the value, one for each of the random variable families in the key, for example:

```py
bm.CompositionalInference({
    (mu, sigma): (bm.SingleSiteNewtonianMonteCarlo(), bm.SingleSiteAncestralMetropolisHastings()),
})
```
which is equivalent to the following, more verbosed syntax with nested `CompositionalInference`:
```py
bm.CompositionalInference({
    (mu, sigma): CompositionalInference({
        mu: bm.SingleSiteNewtonianMonteCarlo(),
        sigma: bm.SingleSiteAncestralMetropolisHastings(),
    }),
})
```

In both of these snippets, we will group all instances of `mu` and `sigma` into a single block, use `SingleSiteNewtonianMonteCarlo()` to update all instances of `mu` and `SingleSiteAncestralMetropolisHastings()` to update all instances of `sigma`. Since `CompositionalInference` itself is just another inference method, you can use it in any places where a inference method is expected.

### Use the default inference method for a block

If you are not planning to change the default inference methods selected by `CompositionalInference` and only want to define a few blocks in your model, you can use Python's `Ellipsis` literal, or equivalently, `...` (three dots), as the value. For example:

```py
bm.CompositionalInference({
    (x,): ...,
    (mu, sigma): ...,
})
```

Similar to the previous example, this is also going to create two blocks for our HMM model. However, since we didn't provide any inference method, `CompositionalInference` will use the default method to update each node in the block instead. Note that this is different from having `Ellipsis` on the left hand side of the dictionary, i.e.,

```py
bm.CompositionalInference({
    ...: bm.SingleSiteNoUTurnSampler(),
})
```
which is used to override the default inference method and *does not* block the nodes automatically (unless you're using a multi-site algorithm, such as `bm.GlobalNoUTurnSampler()`, which always propose values jointly).

To see a live example of running block inference with `CompositionalInference`, check out our [HMM tutorial](../../overview/tutorials/tutorials.md#hidden-markov-model).
