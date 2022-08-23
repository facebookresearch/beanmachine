---
title: Variational Inference
sidebar_label: 'Variational Inference'
slug: '/variational_inference'
---

## Params
A [`Param`](https://beanmachine.org/api/beanmachine.ppl.model.param.html) represents
a variational parameter to be optimized during variational inference.
Use `@bm.param` to decorate an "initialization fuction" which returns a
tensor value to initialize the variational parameter at the start of optimization.

## Variational Worlds
A [`VariationalWorld`](https://beanmachine.org/api/beanmachine.ppl.vi.variational_world.html),
is a sub-class of [`World`](https://beanmachine.org/api/beanmachine.ppl.world.html)
which also contains data on guide distributions and their parameters, specifically:

 - `get_guide_distribution`: given a `RVIdentifier`, returns its corresponding guide distribution
 - `get_param`: given a `RVIdentifier` for a `Param`, returns (possibly initializing if empty) the value of the parameter

__Note__: An implementation detail is that `update_graph` is overriden such that the
guide distribution is automatically used if one is available.

## Gradient Estimators and Divergences
A [`gradient_estimator`](https://beanmachine.org/api/beanmachine.ppl.vi.gradient_estimator.html)
computes a Monte-Carlo (possibly surrogate) objective estimate whose gradients
are used as the training signal.

We structure our VI objective following abstractions introduced in
[f-Divergence Variational Inference](https://arxiv.org/abs/2009.13093), where
`gradient_estimator` takes as input a [discrepancy
function](https://beanmachine.org/api/beanmachine.ppl.vi.discrepancy.html)
corresponding to an $f$-divergence.

## VariationalInfer
The [`VariationalInfer`](https://beanmachine.org/api/beanmachine.ppl.vi.variational_infer.html)
class provides an entrypoint for VI. Model and guide `RVIdentifier`s are associated in the
constructor's `queries_to_guides` argument and optimizater configuration is provided through
a `optimizer` callback. An `infer()` method is provided for easy invocation whereas `step()`
permits more customized interactions (e.g. tensorboard callbacks).


## AutoGuides
Manually defining a guide for each random variable can become tedious.
[`AutoGuide`](https://beanmachine.org/api/beanmachine.ppl.vi.autoguide.html)
provides an initialization strategy for `VariationalInfer` which
automatically defines guides through calling a method 
`get_guide(query: RVIdentifier, distrib: dist.Distribution)` implemented by
subclasses. 

All `AutoGuide`s currently make a mean-field assumption over `RVIdentifiers`:
$$q(x) = \prod_{i \in \text{RVIDs}} q_i(x_i)$$

### ADVI

In Automatic Differentiation Variational Inference (ADVI),
a properly-sized Gaussian is used as a guide to approximate each site:
$$q_i \sim \mathcal{N}(\mu_i, \sigma_i)$$

### MAP

In Maximum A Posteriori (MAP) inference,
a [`Delta`](https://beanmachine.org/api/beanmachine.ppl.distributions.delta.Delta.html)
point estimate is used as the guide for each site:
$$q_i \sim \text{Delta}(\mu_i)$$

