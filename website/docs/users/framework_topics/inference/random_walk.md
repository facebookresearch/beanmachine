# Single-Site Random Walk Metropolis Hastings

Bean Machine offers a module for Random Walk Metropolis-Hastings (RWMH), a simple minimal MCMC inference method. The RWMH module is single-site by default, following the philosophy of most inference methods in Bean Machine, and accordingly block-inference patterns are well supported. RWMH precisely follows the standard Metropolos-Hastings algorithm of sampling a value from a proposal distribution, and then running accept-reject according to the computed ratio of the proposed value. This tutorial describes the proposal mechanism, describes adaptive RWMH, and documents the API for the RWMH module.


# Proposer

The RWMH module has multiple proposers defined on different spaces such as all real numbers, positive real numbers, or intervals of the real numbers. These proposers all have common properties used to propose a new value $Y$ from a current value $X$. The proposal distribution $q(X,Y)$ is constructed to satisfy the following properties $\forall X$:

$$
\mathbb{E}[ q(X, \cdot) ]= X
$$

$$
\mathbb{V} [q(X, \cdot)] = \sigma^2
$$

Note that we haven't defined $\sigma$ yet in this tutorial, but the key property to note is that this is a fixed positive number. $\sigma$ is essentially a hyperparameter of the inference algorithm which must be set by the user, and it must be fixed for the duration of inference. An exception is adaptive inference, a method we describe below which tunes $\sigma$ by essentially looking at the observed data. However, it is important to note that when adaptive RWMH is used, samples drawn during adaptation are not valid, as they potentially violate the balance equations of MCMC.

# Adaptive RWMH

The RWMH module is an exemplar use of the Bean Machine pattern for Adaptive inference, and this is enabled by using the argument `num_adaptive_samples` in the call to `infer()`. This turns on the adaptation module for the specified number of samples, tunes the proposer of the inference method during these timesteps, and records these samples as adaptive samples in the created `MonteCarloSamples` object. The adaptation method used is a single-site adaptation of a well known pattern of doing asympototically smaller steps on the value $log(\sigma)$. Details of this method can be found at http://www.stats.ox.ac.uk/~evans/CDT/Adaptive.pdf.

# API

The API can be called as follows. Note that $\sigma$ is denoted as `step_size` in the constructor.


```
from beanmachine.ppl import SingleSiteRandomWalk

monte_carlo_samples = SingleSiteRandomWalk(
  step_size = 2.0,
).infer(
  queries,
  observations,
  num_adapt_steps = 1000,
  num_steps = 200,
)

```

If desired, `step_size` does not need to be set, and it will be initialized to the default initial value `1.0`. Either way, if `num_adapt_steps>0` is set, then `step_size` will be changed after inference begins.

```
from beanmachine.ppl import SingleSiteRandomWalk

monte_carlo_samples = SingleSiteRandomWalk().infer(
  queries,
  observations,
  num_adapt_steps = 1000,
  num_steps = 200,
)

```
