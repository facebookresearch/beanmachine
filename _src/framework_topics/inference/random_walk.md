---
id: random_walk
title: 'Single-Site Random Walk Metropolis-Hastings'
sidebar_label: 'Single-Site Random Walk MH'
slug: '/random_walk'
---

Random Walk Metropolis-Hastings is a simple, minimal MCMC inference method. Random Walk Metropolis-Hastings is single-site by default, following the philosophy of most inference methods in Bean Machine, and accordingly multi-site inference patterns are well supported. Random Walk Metropolis-Hastings follows the standard Metropolis-Hastings algorithm of sampling a value from a proposal distribution, and then running accept-reject according to the computed ratio of the proposed value. This is further detailed in the docs for [Ancestral Metropolis-Hastings](ancestral_metropolis_hastings.md). This tutorial describes the proposal mechanism, describes adaptive Random Walk Metropolis-Hastings, and documents the API for the Random Walk Metropolis-Hastings algorithm.

## Algorithm

Random Walk Metropolis-Hastings works on a single-site basis by proposing new values for a random variable that are close to the current value according to some sense of distance. As such, it is only defined for continuous random variables. The exact distance that a proposed value is from the current value is defined by the _proposal distribution_, and is a parameter that can be provided when configuring the inference method. For discrete random variables, a similar effect may be achieved, but [custom proposers](../custom_proposers/custom_proposers.md) must be used instead.

The Random Walk Metropolis-Hastings algorithm has multiple proposers defined on different spaces such as all real numbers, positive real numbers, or intervals of the real numbers. These proposers all have common properties used to propose a new value $x^\prime$ from a current value $x$. The proposal distribution $q(x,x^\prime)$ is constructed to satisfy the following properties:

$$
  \begin{aligned}
    \mathbb{E}[q(x, \cdot)] &= x \\
    \mathbb{V} [q(x, \cdot)] &= \sigma^2
  \end{aligned}
$$

$\sigma$ is the parameter that may be provided as a parameter when configuring the inference method, and it must be a fixed positive number. Larger values of $\sigma$ will cause the inference method to explore more non-local values for $X$. This may be good for faster exploration of the posterior, but it may cause lower probability values to get proposed (and therefore rejected) as a result.

## Adaptive Random Walk Metropolis-Hastings

Selecting a good $\sigma$ value is important for efficient posterior exploration. However, it is often challenging for a user to select a good $\sigma$ value, as it requires a nuanced understanding of the posterior space. Consequently, Bean Machine provides an adaptive version of Random Walk Metropolis-Hastings, in which the inference engine automatically tunes the value of $\sigma$ during the first few samples of inference (known as the adaptation period).

The Random Walk Metropolis-Hastings algorithm is an exemplar use of the Bean Machine pattern for Adaptive inference, and this is enabled by using the argument `num_adaptive_samples` in the call to `infer()`. This causes Bean Machine to run an adaptation phase at the beginning of inference for the provided number of samples. During this phase, Bean Machine will internally tweak values of $\sigma$ in order to find the largest value that still results in a relatively low number of rejected proposals. Technically speaking, Random Walk adaptation will attempt to achieve an amortized acceptance rate of 0.234. How this value is chosen as the optimal acceptance rate is detailed in [Optional Scaling and Adaptive Markov Chain Monte Carlo](http://www.stats.ox.ac.uk/~evans/CDT/Adaptive.pdf).

Please note that samples taken during adaptation are not valid posterior samples, and so will not be shown by default when using the `MonteCarloSamples` object returned from inference.

## Usage

The following code snippet illustrates how to use the inference method. Here, `step_size` represents $\sigma$ from the algorithm above.

```py
samples = bm.SingleSiteRandomWalk(
  step_size = 2.0,
).infer(
  queries,
  observations,
  num_adapt_steps = 1000,
  num_steps = 200,
)
```

If desired, `step_size` does not need to be set, and it will be initialized to the default initial value `1.0`. Either way, if `num_adapt_steps > 0` is set, then `step_size` will be changed after inference begins.

```py
samples = bm.SingleSiteRandomWalk().infer(
  queries,
  observations,
  num_adapt_steps = 1000,
  num_steps = 200,
)
```

The parameters to `infer` are described below:

| Name | Usage
| --- | ---
| `queries` | A `List` of `@bm.random_variable` targets to fit posterior distributions for.
| `observations` | The `Dict` of observations. Each key is a random variable, and its value is the observed value for that random variable.
| `num_samples` | Number of samples to build up distributions for the values listed in `queries`.
| `num_chains` | Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.
