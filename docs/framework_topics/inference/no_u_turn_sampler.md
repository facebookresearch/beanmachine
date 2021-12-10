---
id: no_u_turn_sampler
title: No-U-Turn Sampler
sidebar_label: 'No-U-Turn Sampler'
slug: '/no_u_turn_sampler'
---

The No-U-Turn Samplers (NUTS) algorithm is an inference algorithm for differentiable random variables which uses Hamiltonian dynamics. It is an extension to the Hamiltonian Monte Carlo (HMC) inference algorithm.

:::tip

If you haven't already read the docs on [Hamiltonian Monte Carlo](hamiltonian_monte_carlo.md), please read those first.

:::

## Algorithm

The goal for NUTS is to automate the selection of an appropriate path length $\lambda$ for HMC inference. It extends HMC by allowing the simulation of steps backwards in time during the leapfrog step. It also uses a smart simulation algorithm that can choose a path length heuristically, so that the proposed value tends to have a low correlation with the current value.

NUTS dynamically determines when the path starts looping backwards. In combination with the improvements from Adaptive HMC, this allow Bean Machine to automatically find the best step size and path length without requiring any user-tuned parameters.

NUTS decides on an optimal path length by building a binary tree where each path through the binary tree represents the trajectory of a sample. Each node at depth $j$ represents simulating $2^j$ steps forwards or backwards. This binary tree is adaptively grown until either hitting a pre-specified max depth size, or until the algorithm starts proposing samples with too low probabilities due to discretization errors.

The full NUTS algorithm description is quite involved. We recommend you check out [Hoffman & Gelman, 2011](https://arxiv.org/pdf/1111.4246.pdf) to learn more.

## Usage

Bean Machine provides a single-site version of NUTS that only updates one variable at a time:

```py
bm.SingleSiteNoUTurnSampler(
    use_dense_mass_matrix=True,
).infer(
    queries,
    observations,
    num_samples,
    num_chains,
    num_adaptive_samples=1000,
)
```

:::caution

**Make sure to set `num_adaptive_samples` when using adaptive HMC!** If you forget to set `num_adaptive_samples`, no adaptation will occur.

:::

Bean Machine also provides a multi-site version of NUTS that updates all variables in your model at the same time. This is only appropriate for models that are comprised of only continuous random variables.

```py
bm.GlobalNoUTurnSampler(
    max_tree_depth=10,
    max_delta_energy=1000.0,
    initial_step_size=1.0,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    multinomial_sampling=True,
    target_accept_prob=0.8,
).infer(
    queries,
    observations,
    num_samples,
    num_chains,
    num_adaptive_samples=1000,
)
```

The `GlobalNoUTurnSampler` has all the acceptance step size, covariance matrix, and acceptance probability tuning arguments of `GlobalHamiltonianMonteCarlo` as well as a few more parameters related to tuning the path length. While there are many optional parameters for this inference method, in practice, the parameters you are most likely to modify are `target_accept_prob` and `max_tree_depth`. When dealing with posteriors where the probability density has a more complicated shape, we benefit from taking smaller steps. Setting `target_accept_prob` to a higher value like `0.9` will lead to a more careful exploration of the space using smaller step sizes while still benefiting from some tuning of that step size. Since we will be taking smaller steps, we need to compensate by having a larger path length. This is accomplished by increasing `max_tree_depth`. Otherwise, using the defaults provided is highly recommended.

A more complete explanation of parameters to `GlobalNoUTurnSampler` are provided below:

| Name | Usage
| --- | ---
| `max_tree_depth` | The maximum depth of the binary tree used to simulate leapfrog steps forwards and backwards in time.
| `max_delta_energy` | This is the lowest probability moves that NUTS will consider. Once most new samples have a lower probability, NUTS will stop its leapfrog steps. This should be interpreted as a negative log probability and is designed to be fairly conservative.
| `initial_step_size` | The initial step size $\epsilon$ used in adaptive HMC. This value is simply the step size if tuning is disabled.
| `multinomial_sampling` | Lets us decide between a faster multinomial sampler for the trajectory or the slice sampler described in the [original paper](https://arxiv.org/pdf/1111.4246.pdf). The option is useful for fairly comparing against other NUTS implementations.
| `target_accept_prob` | Indicates the acceptance probability which should be targeted by the step size tuning algorithm. While the optimal value is 65.1%, higher values have been show to be more robust leading to a default of 0.8.

The parameters to `infer` are described below:

| Name | Usage
| --- | ---
| `queries` | A `List` of `@bm.random_variable` targets to fit posterior distributions for.
| `observations` | The `Dict` of observations. Each key is a random variable, and its value is the observed value for that random variable.
| `num_samples` | Number of samples to build up distributions for the values listed in `queries`.
| `num_chains` | Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.


---

Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." _J. Mach. Learn. Res._ 15.1 (2014): 1593-1623.
