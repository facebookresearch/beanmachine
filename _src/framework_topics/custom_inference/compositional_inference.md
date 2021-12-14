---
id: compositional_inference
title: 'Compositional Inference'
sidebar_label: 'Compositional Inference'
slug: '/compositional_inference'
---

Sometimes it might be hard to pick a single algorithm that performs well on the entire model. For example, while gradient-based algorithms such as [No-U-Turn Sampler](../inference/no_u_turn_sampler.md) and [Newtonian Monte Carlo](../inference/newtonian_monte_carlo.md) generally yield high number of effective samples, they can only handle random variables with continuous support. On the other hand, while single site algorithms make it easier to update high-dimensional models by proposing only one node at a time, they might have trouble updating models with highly correlated random variables. Fortunately, Bean Machine supports composable inference through the `CompositionalInference` class, which allows us to use different inference methods to update different subset of nodes and to "block" multiple nodes together so that they are accepted/rejected jointly by a single Metropolis-Hastings step. In this doc, we will cover the basics of `CompositionalInference` and how to mix-and-match different inference algorithms. To learn about how to do "block inference" with `CompositionalInference`, see [Block Inference](block_inference.md).

## Default Inference Methods

By default, Compositional Inference will pick a single site algorithm to update each of the latent variable in the model based on its support:

| Support | Algorithm
| --- | ---
| real | `SingleSiteNewtonianMonteCarlo` (real space proposer)
| greater than | `SingleSiteNewtonianMonteCarlo` (half space proposer)
| simplex |`SingleSiteNewtonianMonteCarlo` (simplex space proposer)
| finite discrete | `SingleSiteUniformMetropolisHastings`
| everything else | `SingleSiteAncestralMetropolisHastings`

To run `CompositionalInference` with these default inference methods, simply leave the inference argument empty:

```py
CompositionalInference().infer(
    queries,
    observations,
    num_samples,
    num_chains,
    num_adaptive_samples,
)
```

The parameters to `infer` are described below:

| Name | Usage
| --- | ---
| `queries` | A `List` of `@bm.random_variable` targets to fit posterior distributions for.
| `observations` | The `Dict` of observations. Each key is a random variable, and its value is the observed value for that random variable.
| `num_samples` | Number of samples to build up distributions for the values listed in `queries`.
| `num_chains` | Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.
| `num_adaptive_samples` | The integer number of samples to spend before `num_samples` on tuning the inference algorithm for the `queries`.


## Configuring Your Own Inference

`CompositionalInference` also takes an optional dictionary, namely, the `inference_dict`, which can be used to override the default behavior.
In the following sections, assume that we're working with a toy model with three random variable families, `foo`, `bar`, and `baz`:

```py
@bm.random_variable
def foo(i):
    return dist.Beta(2.0, 2.0)

@bm.random_variable
def bar(i):
    return dist.Bernoulli(foo(i))

@bm.random_variable
def baz(j):
    return dist.Normal(0.0, 1.0)
```

### Choosing different inference methods for different random variable families

To select an inference algorithm for a particular random variable family, pass the random variable family as the key and an instance of the inference algorithm as value.
For example, the following snippet tells `CompositionalInference` to use `SingleSiteNoUTurnSampler()` to update all instances of `foo` and `SingleSiteHamiltonianMonteCarlo(1.0)` to update all instance of `baz`.
Nodes that are not specified, such as instances of `bar`, will fall back to the default inference methods mentioned above.

```py
bm.CompositionalInference({
    foo: bm.SingleSiteNoUTurnSampler(),
    baz: bm.SingleSiteHamiltonianMonteCarlo(trajectory_length=1.0),
}).infer(**args) # same parameters as shown above
```

You may notice that we are using what we referred to as "random variable families" like `foo` as keys, which are essentially functions that generates the random variables, instead of using instances of random variables such as `foo(0)` and `foo(1)`. This is because often times the number of random variable is not known ahead of time until an inference starts with some data (you can even have unbounded number of nodes in some model). By using random variable family in the config, we no longer need to explicitly spell out all every instance in a huge dictionary.

### Overriding default inference method

If your model has a large number of nodes and you want to override the default inference method for all of them without listing them all, you can use Python's `Ellipsis` literal, or equivalently, `...` (three dots), as a key to specify the default inference method for nodes that are not specified in the dictionary. For example, the following code snippet tells `CompositionalInference` to use `SingleSiteUniformMetropolisHastings()` to update all instances of `bar` (which are discrete), and use `SingleSiteNoUTurnSampler()` to update the rest of nodes (which are all continuous).

```py
bm.CompositionalInference({
    bar: bm.SingleSiteUniformMetropolisHastings(),
    ...: bm.SingleSiteNoUTurnSampler(),
}).infer(**args) # same parameters as shown above
```

Bean Machine provides a great variety of inference methods under [`beanmachine.ppl.inference`](https://beanmachine.org/api/beanmachine.ppl.inference.html) that can be used with the `CompositionalInference` API. To learn more about what else can be done with `CompositionalInference`, see [Block Inference](block_inference.md).
