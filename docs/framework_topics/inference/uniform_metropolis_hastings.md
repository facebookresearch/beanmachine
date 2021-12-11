---
id: uniform_metropolis_hastings
title: 'Single-Site Uniform Metropolis-Hastings'
sidebar_label: 'Single-Site Uniform MH'
slug: '/uniform_metropolis_hastings'
---

Single-Site Uniform Metropolis-Hastings is used to infer over variables that have discrete support, for example random variables with Bernoulli and Categorical distributions. It is overall very similar to Ancestral Metropolis-Hastings. However, it is designed so that it will even explore discrete samples that are unlikely under the prior distribution.

## Algorithm

The Single-Site Uniform Sampler works very similarly to [Single-Site Ancestral Metropolis-Hastings](ancestral_metropolis_hastings.md). In fact, the only difference arises in Step 1 of that inference method's Algorithm; i.e, in the way that this sampler proposes a new value. The remaining steps are unchanged.

In Single-Site Uniform Metropolis-Hastings, for random variables with discrete support, instead of sampling from the prior, the proposer samples from a distribution which assigns equal probability across all values in support (hence the name, uniform). However, the likelihood of this sample _is_ accounted for when computing the [Metropolis acceptance probability](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Formal_derivation). Thus, even though improbable values may be _proposed_ more than indicated by the prior, they will not be _accepted_ more often than they should according to the posterior.

At first appearance, this sounds undesirable -- why sample an unlikely value in the first place? This arises from the fact that the prior distribution may not be a good reflection of the posterior distribution for a given discrete random variable. A particular value that is unlikely under the prior may, in fact, be quite likely under the posterior. Uniform Metropolis-Hastings ensures that those values have the opportunity to be sampled, and thus can increase sampling efficiency for many problems where the posterior is distant from the prior.

Please note that, if you use this inference method for continuous random variables, it will fall back to Single-Site Ancestral Metropolis-Hastings.

## Usage

The following code snippet illustrates how to use the inference method.

```py
samples = bm.SingleSiteUniformMetropolisHastings().infer(
    queries,
    observations,
    num_samples,
    num_chains,
)
```

The parameters to `infer` are described below:

| Name | Usage
| --- | ---
| `queries` | A `List` of `@bm.random_variable` targets to fit posterior distributions for.
| `observations` | The `Dict` of observations. Each key is a random variable, and its value is the observed value for that random variable.
| `num_samples` | Number of samples to build up distributions for the values listed in `queries`.
| `num_chains` | Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.
