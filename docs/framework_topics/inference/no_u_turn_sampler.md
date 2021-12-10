---
id: no_u_turn_sampler
title: No-U-Turn Sampler
sidebar_label: 'No-U-Turn Sampler'
slug: '/no_u_turn_sampler'
---

<!-- TODO(tingley): Review and optimize this doc page. -->

The No-U-Turn Samplers (NUTS) dynamically determines when the path starts looping backwards. In combination with the improvements from Adaptive HMC, this allow Bean Machine to automatically find the best step size and path length without requiring any user-tuned parameters.

NUTS decides on an optimal path length by building a binary tree where each path through the binary tree represents the trajectory of our particle. Each node at depth $j$ representing going $2^j$ steps forwards or backwards. This binary tree is adaptively grown until either hitting a max depth size, or due to discretization error we start proposing moves with too low of a probability.

Bean Machine comes with two variants of NUTS, a single-site implementation that only updates one variable at a time and another that updates
all the variables at once.

```py
nuts = bm.SingleSiteNoUTurnSampler(use_dense_mass_matrix = True)
nuts.infer(queries, observations, num_samples, num_chains, num_adaptive_samples=1000)
```

```py
nuts = bm.GlobalNoUTurnSampler(
    max_tree_depth = 10,
    max_delta_energy = 1000.0,
    initial_step_size = 1.0,
    adapt_step_size = True,
    adapt_mass_matrix = True,
    multinomial_sampling = True,
    target_accept_prob = 0.8,
)

nuts.infer(queries, observations, num_samples, num_chains, num_adaptive_samples=1000);
```

The `GlobalNoUTurnSampler` has all the acceptance step size, covariance matrix and acceptance probability tuning arguments of `GlobalHamiltonianMonteCarlo` as well as a few more related to tuning the path length. We set the maximum size of the binary tree using `max_tree_depth`. The `max_delta_energy` is roughly the lowest probability moves we will consider. This should be interpreted as a negative log probability and is designed to be fairly conservative. The initial step size is set by `initial_step_size` where this values is simply the step size if tuning is disabled. The argument `multinomial_sampling` lets us decide between a faster multinomial sampler for the trajectory or the slice sampler in the original paper. The option is useful for fairly comparing against other NUTS implementations. The `target_accept_prob` argument indicates the acceptance probability which should be targeted by the step size tuning algorithm. While the optimal value is 65.1% higher values have been show to be more robust leading to a default of 0.8.

In practice, the parameters you are most likely to modify are `target_accept_prob` and `max_tree_depth`. When dealing with posteriors where the probability density has a more complicated shape, we benefit from taking smaller steps. Setting `target_accept_prob` to a higher value like `0.9` will need to a more careful exploration of the space using smaller step sizes while still benefiting from some tuning of that step size. Since we will be taking smaller steps we need to compensate by having a larger path length. This is accomplished by increasing `max_tree_depth`. Otherwise, using the defaults provided is highly recommended.
