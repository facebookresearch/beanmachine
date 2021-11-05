## Hamiltonian Monte Carlo and Variations

### Hamiltonian Monte Carlo

Hamiltonian Monte Carlo (HMC) is a sampling algorithm for differentiable random variables which uses Hamiltonian dynamics. By randomly drawing a momentum for the kinetic energy and treating the true posterior as the potential energy, HMC is able to simulate trajectories which explore the space. Intuitively, this can be viewed as starting with a marble at a point inside a bowl, flicking the marble in a random direction, and then following the marble as it rolls around. The position of the marble represents the sample, the flick represents the momentum, and the shape of the bowl in combination with the force of gravity represents our true posterior.

#### Algorithm Overview
More formally, HMC applies Hamiltonian dynamics to the state space by introducing an auxiliary momentum variable $p$. The Hamiltonian can then be written as
$$H(q, p) = U(q) + K(p)$$
where $U$ and $K$ represent the potential and kinetic energy respectively.

The potential energy represents the shape of the posterior distribution, and is defined by
$$U(q) = -\log[\pi(q)L(q\mid D)]$$
The kinetic energy is defined using the momentum variable $p$ as well as the covariance matrix $\Sigma$.
$$K(p) = p^T\Sigma p/2$$
We can then simulate the trajectory using the following forms of Hamilton's equations:
$$\frac{dq_i}{dt} = [\Sigma p]_i$$
$$\frac{dp_i}{dt} = -\frac{\partial U}{\partial q_i}$$

However, because these equations cannot be directly computed, Bean Machine uses the leapfrog method to approximate the trajectory. For leapfrog step at time $t$ of size $\epsilon$, we take a half-step for momentum and a full-step for position using the updated momentum, and finally another half step for the momentum.
$$p_i(t + \epsilon/2) = p - (\epsilon/2)\frac{\partial U}{\partial q_i}(q(t))$$
$$q_i(t + \epsilon) = q_i(t) + \epsilon \Sigma p_i(t + \epsilon/2)$$
$$p_i(t + \epsilon) = p_i(t + \epsilon/2) - (\epsilon/2)\frac{\partial U}{\partial q_i}(q(t + \epsilon))$$

With target path length $\lambda$ and step size $\epsilon$, the number of leapfrog steps is calculated by $\lceil\lambda / \epsilon\rceil$. The final sample for $q$ is chosen by the value of $q$ after the last leapfrog step.

Due to the discretization, the resulting trajectory will contain numerical errors. To account for the error, a Metropolis Hastings accept / reject will be applied and the value for $q$ will be updated depending accordingly.

#### HMC in Bean Machine

In Bean Machine, inference using HMC can be specified as an inference method for all variables in the model

```py
hmc = bm.SingleSiteHamiltonianMonteCarlo(path_length=1.0, step_size=0.1);
```
or through compositional inference to select HMC for specific variables

```py
hmc = bm.CompositionalInference(
  {x: SingleSiteHamiltonianMonteCarloProposer(path_length=1.0, step_size=0.1)}
);
```
The parameters for HMC have to be carefully selected, as different parameters may lead to vastly different behavior.
* Path length:
Because the samples should be minimally correlated, it is ideal to follow the trajectory for long path lengths. However, distributions may have periodic behavior, and long path lengths may waste computation. The ideal path length is the minimal path length where the samples have low correlation.
* Step Size:
If the Hamiltonian equations were followed perfectly, all samples would be accepted. However, error is introduced into the system during the discretization of Hamilton equations. The larger the step size, the worse the final approximation will be; however, if the steps are too small, the number of steps needed as well as the overall runtime of the algorithm will increase.

### Adaptive Hamiltonian Monte Carlo

Adaptive HMC requires an adaptive phase, where we use the HMC algorithm to generate samples while tuning its own step size and covariance matrix. Adaptive HMC provides two main improvements over HMC.

* Users do not have to specify a step size:
    The ideal acceptance rate for HMC is 65%, where step size is aggressive enough to minimize computation costs, but small enough that the discretization does not introduce too much error into the system.
    During the adaptive phase, the step size is adjusted based on the acceptance probability of the samples. If the acceptance rate is above the ideal acceptance rate, then Bean Machine is being too careful and discretizing in steps that are too small; therefore, the step size should be increased. If the acceptance rate is too low, then the step size should be decreased. We follow the Robbins Munro stochastic approximation method, where earlier iterations within the adaptive phase have a larger influence over the step size than later iterations.
* HMC can take different step sizes in different dimensions:
    During the adaptive phase, tune the momentum in each dimension is tuned depending on the covariance of the previously accepted samples. The amount of momentum used is controlled by the covariance matrix, which allows HMC to move through the dimensions at different rates.
    The estimated covariance matrix is adjusted based on the covariance of the samples. Since the ideal covariance matrix is the true covariance, we can approximate this during the adaptive phase by using the covariance of the samples.

Once the adaptive phase ends, we no longer update our parameters, and the original HMC algorithm is used to generate new samples. Since the samples generated during the adaptive phase use untuned parameters, they may not be of the highest quality and are not returned by default.

Adaptive HMC will be used when no step size is specified.
```py
hmc = bm.SingleSiteHamiltonianMonteCarlo(path_length=1.0);
```
During inference, make sure to specify the number of adaptive samples, else no adaptation will occur.
```py
hmc.infer(queries, observations, num_samples, num_chains, num_adaptive_samples=1000)
```


### No-U-Turn Sampler (will be detailed later as implementation progresses)

The No-U-Turn Samplers (NUTS) dynamically determines when the path starts looping backwards. In combination with the improvements from Adaptive HMC, this allow Bean Machine to automatically find the best step size and path length without requiring any user-tuned parameters.

Bean Machine NUTS can be used as follows

```py
nuts = bm.SingleSiteNoUTurnSampler();
```
