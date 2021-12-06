---
id: uniform_metropolis_hastings
title: 'Single-Site Uniform Metropolis-Hastings'
sidebar_label: 'Single-Site Uniform MH'
slug: '/uniform_metropolis_hastings'
---

Single-Site Uniform Metropolis-Hastings is used to infer over variables that have discrete support, for example random variables with Bernoulli and Categorical distributions. The Single-Site Uniform Sampler works very similarly to the Single-Site Ancestral Metropolis-Hastings. In fact, the only difference arises in Step 1, i.e, in the way that this sampler proposes a new value. Steps 2-4 are the same.

In Single-Site Uniform MH, for random variables with discrete support, instead of sampling from the prior, the proposer samples from a distribution which assigns equal probability across all values in support (hence the name, uniform). For any random variables with continuous support, this inference method resorts to Single-Site Ancestral MH.

Here is an example of how to use Single-Site Uniform Metropolis-Hastings to perform inference in Bean Machine.

```
import beanmachine.ppl as bm

mh = bm.SingleSiteUniformMetropolisHastings()
coin_samples = mh.infer(queries, observations, num_samples, num_chains, run_in_parallel)
```

*TODO Remove the explanation of parameters below; they have already been explained in `overview/inference/inference.md`, with the exception of `run_in_paralell`*

```queries ```: List of random variables that we want to get posterior samples for
```observations```: Dict, where key is the random variable, and value is the value of the random variable
```num_samples```: number of samples to run inference for
```num_chains```: number of chains to run inference for
```run_in_parallel```: True if you want the chains to run in parallel
