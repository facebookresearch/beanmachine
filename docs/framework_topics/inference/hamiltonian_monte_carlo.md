---
id: hamiltonian_monte_carlo
title: Hamiltonian Monte Carlo
sidebar_label: 'Hamiltonian Monte Carlo'
slug: '/hamiltonian_monte_carlo'
---

Hamiltonian Monte Carlo (HMC) is a sampling algorithm for differentiable random variables which uses Hamiltonian dynamics. By randomly drawing a momentum for the kinetic energy and treating the true posterior as the potential energy, HMC is able to simulate trajectories which explore the space. Intuitively, this can be viewed as starting with a marble at a point inside a bowl, flicking the marble in a random direction, and then following the marble as it rolls around. The position of the marble represents the sample, the flick represents the momentum, and the shape of the bowl in combination with the force of gravity represents our true posterior.

## Hamiltonian dynamics

HMC applies Hamiltonian dynamics to explore the state space. This means that we think of the posterior surface as having some potential energy, proportional to the posterior's negative log likelihood at some particular set of parameter assignments. We use that potential energy, along with kinetic energy injected in the form of a random momentum "kick", to traverse the surface according to the laws of physics.

Below, we'll use $q$ to represent the current set of parameter assignments, and $p$ to represent a random momentum factor injected into the system by the inference method. The Hamiltonian physics alluded to above can be written as

$$
  H(q, p) = U(q) + K(p),
$$

where $U$ and $K$ represent the potential and kinetic energy respectively.

The potential energy represents the shape of the posterior distribution, and is defined by

$$
    U(q) = -\log[\pi(q)L(q\mid D)].
$$

Here, $L(q\mid D)$ is the likelihood of the model evaluated at the parameter assignments $q$, conditioned on the observed dataset $D$.

The potential energy can be evaluated directly in Bean Machine as a function of the model's posterior probability at the current parameter values.

The kinetic energy is defined using the momentum variable $p$ as well as an inference-method-defined covariance matrix $\Sigma$,

$$
    K(p) = p^T\Sigma p/2.
$$

The values for $p$ and $\Sigma$ are provided by the framework. Bean Machine samples from a Normal distribution scaled to the appropriate posterior size for $p$. Bean Machine can optionally estimate an appropriate constant value for $\Sigma$ by measuring correlations between parameters during an adaptation phase of inference.

We can then simulate the trajectory using the following forms of Hamilton's equations:

$$
  \begin{aligned}
    \frac{dq_i}{dt} &= [\Sigma p]_i\\
    \frac{dp_i}{dt} &= -\frac{\partial U}{\partial q_i}
  \end{aligned}
$$

Bean Machine can compute $\frac{\partial U}{\partial q_i}$ using autograd.

The goal of these equations is to determine where the new sample will come to rest after a framework-specified path length $\lambda$. Both the potential energy (from the posterior's likelihood) and the kinetic energy (from the injected "kick") will interact with the sampled value to influence how it travels over the posterior surface.

## Approximating Hamiltonian dynamics

Unfortunately, this system of differential equations cannot be analytically computed. Instead, Bean Machine discretizes these equations based on a discrete time step $t$, and simulates how they influence each other in this discrete setting for a framework-specified path length $\lambda$. This discrete simulation is referred to as the "leapfrog" algorithm. Each simulated time step is referred to as a "leapfrog step".

At a high level, leapfrog works like this:

  1. Choose momentum ($p$) and covariance ($\Sigma$) values to use for the simulation.
  2. Simulate a small time step of the momentum's influence on the sample.
  3. Simulate a small time step of the potential energy's influence on the sample.
  4. Repeat for the framework-specified path length $\lambda$.

We'll represent the above description mathematically. For leapfrog step at time $t$ of size $\epsilon$, we take a half-step for momentum and a full-step for potential energy using the updated momentum, and finally another half step for the momentum.

$$
  \begin{aligned}
    p_i(t + \epsilon/2) &= p - (\epsilon/2)\frac{\partial U}{\partial q_i}(q(t))\\
    q_i(t + \epsilon) &= q_i(t) + \epsilon \Sigma p_i(t + \epsilon/2)\\
    p_i(t + \epsilon) &= p_i(t + \epsilon/2) - (\epsilon/2)\frac{\partial U}{\partial q_i}(q(t + \epsilon))
  \end{aligned}
$$

This process is repeated until we have achieved the framework-specific distance $\lambda$. Thus, the number of leapfrog steps is calculated by $\lceil\lambda / \epsilon\rceil$. The final sample for $q$ is chosen by the value of $q$ after the last leapfrog step.

Due to the discretization, the resulting trajectory will contain numerical errors. To account for the error, a [Metropolis acceptance probability](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Formal_derivation) will be used to determine whether to accept the proposed value for $q$.

HMC's performance is quite sensitive to the hyperparameters used. They are as follows:

* **Path length, $\lambda$.**  Because the samples should be minimally correlated, it is ideal to follow the trajectory for long path lengths. However, distributions may have periodic behavior, and long path lengths may waste computation. The ideal path length is the minimal path length where the starting and ending samples have low correlation.
* **Step size, $\epsilon$.**  If the Hamiltonian equations were followed exactly (not approximated), all samples would be accepted. However, error is introduced into the system during the discretization of Hamilton's equations. The larger the step size, the worse the final approximation will be; however, if the steps are too small, the number of steps needed as well as the overall runtime of the algorithm will increase.
* **Covariance (mass) matrix, $\Sigma$**.  If there is significant correlations between the variables and the covariance matrix is not properly tuned, we will need to make smaller steps to get samples that are likely to be accepted.

The optimal acceptance rate for HMC, as derived by Neal (2011), is 0.65. It is desirable to tune these parameters so that they acheive this acceptance rate on average.

## Adaptive Hamiltonian Monte Carlo

Due to the challenge of selecting good hyperparameters, Bean Machine provides extensions to HMC to help choose appropriate values.

One such extension is called Adaptive Hamiltonian Monte Carlo. Adaptive HMC Adaptive HMC requires an [adaptive phase](../programmable_inference/adaptive_inference.md), where Bean Machine uses the HMC algorithm to generate samples while tuning HMC's  step size and covariance matrix. Adaptive HMC provides two main improvements over HMC, outlined below.

**Users do not have to specify a step size.**  During the adaptive phase, the step size is adjusted in order to achieve the optimal acceptance rate of 0.65 on average. If the acceptance rate is above optimal, then Bean Machine is being too careful and discretizing in steps that are too small; therefore, the step size should be increased. If the acceptance rate is too low, then the step size should be decreased. We follow the [Robbins-Monro stochastic approximation method](https://en.wikipedia.org/wiki/Stochastic_approximation#Robbins%E2%80%93Monro_algorithm), where earlier iterations within the adaptive phase have a larger influence over the step size than later iterations.

**HMC can take different step sizes in different dimensions.**  During the adaptive phase, the momentum in each dimension is tuned depending on the covariance of the previously-accepted samples. The amount of momentum used is controlled by the covariance matrix, which allows HMC to move adjust the dimensions at different rates. The estimated covariance matrix is adjusted based on the covariance of the samples. Since the ideal covariance matrix is the true covariance, we can approximate this during the adaptive phase by using the covariance of the samples.

Once the adaptive phase ends, we no longer update our parameters, and the original HMC algorithm is used to generate new samples. Since the samples generated during the adaptive phase use untuned parameters, they may not be of the highest quality and are not returned by default.

## Tuning the path length

While adaptive HMC can effectively tun the step size and covariance matrix, Bean Machine relies on a separate algorithm for tuning the path length $\lambda$. This algorithm is called the No-U-Turn Sampler, and has its own documentation page, [no_u_turn_sampler.md](no_u_turn_sampler.md).

## Usage

In Bean Machine, inference using HMC can be specified as an inference method for all variables in the model:

```py
bm.SingleSiteHamiltonianMonteCarlo(
    path_length=1.0,
    step_size=0.1,
).infer(
    queries,
    observations,
    num_samples,
    num_chains,
)
```

[`CompositionalInference`](../programmable_inference/compositional_inference.md)can alternatively be used to select HMC for specific variables:

```py
bm.CompositionalInference({
    x: SingleSiteHamiltonianMonteCarloProposer(
        path_length=1.0,
        step_size=0.1,
    ),
}).infer(
    # Same arguments as above snippet
)
```

All the above will only update one random variable at a time per iteration. To resample all variables at once, use:

```py
bm.GlobalHamiltonianMonteCarlo(
    trajectory_length=1.0,
    step_size=0.1,
).infer(
    # Same arguments as above snippet
)
```

Adaptive HMC will be used automatically when no step size is specified:

```py
bm.SingleSiteHamiltonianMonteCarlo(
    path_length=1.0,
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

The global variant for adaptive HMC, `GlobalHamiltonianMonteCarlo`, comes with a few more options for tuning the step size and acceptance probability:

```py
bm.GlobalHamiltonianMonteCarlo(
    trajectory_length=1.0,
    initial_step_size=1.0,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    target_accept_prob=0.8,
).infer(
    queries,
    observations,
    num_samples,
    num_chains,
    num_adaptive_samples=1000,
)
```

These arguments allow us to decide if we want to tune the step size `adjust_step_size` or covariance matrix `adapt_mass_matrix`.  The `target_accept_prob` argument indicates the acceptance probability which should be targeted by the step size tuning algorithm. While the optimal value is 65.1%, higher values have been show to be more robust. As a result, Bean Machine targets an acceptance rate of 0.8 by default. **Again, do not forget to specify `num_adaptive_samples`, or no adaptation will occur.**

The parameters to `infer` are described below:

| Name | Usage
| --- | ---
| `queries` | A `List` of `@bm.random_variable` targets to fit posterior distributions for.
| `observations` | The `Dict` of observations. Each key is a random variable, and its value is the observed value for that random variable.
| `num_samples` | Number of samples to build up distributions for the values listed in `queries`.
| `num_chains` | Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.


---

Neal, Radford M. "MCMC using Hamiltonian dynamics." _Handbook of markov chain monte carlo_ 2.11 (2011): 2.

Herbert Robbins. Sutton Monro. "A Stochastic Approximation Method." _Ann. Math. Statist._ 22 (3) 400 - 407, September, 1951. https://doi.org/10.1214/aoms/1177729586
