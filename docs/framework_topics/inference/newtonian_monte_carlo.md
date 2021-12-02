---
id: newtonian_monte_carlo
title: 'Newtonian Monte Carlo'
sidebar_label: 'Newtonian Monte Carlo'
slug: '/newtonian_monte_carlo'
---

Newtonian Monte Carlo (NMC, Arora et al., 2020) is a second-order gradient-based Markov chain Monte Carlo (MCMC) algorithm that uses the first- and second-order gradients to propose a new value for a random variable.

To understand the idea behind NMC, we revisit two other common MCMC algorithms, [Random Walk](framework_topics/inference/random_walk.md) and Metropolis-adjusted Langevin algorithm (MALA). In Random Walk, we propose the next parameter value $\theta\in\R^k$ for our model using a simple Normal distribution centered at the current value: $q(. \mid \theta) = \mathcal{N}(\theta, \epsilon^2I_k)$. This Normal distribution has covariance $\epsilon^2I_k$ where $\epsilon$ is the user-specified step size and $I_k$ ​is a $k \times k$ identity matrix (Metropolis et al., 1953). In cases where the target density, $\pi(\theta)$, is differentiable, we can use the proposal distribution proposed by MALA which is $q(.|\theta)=\mathcal{N}(\theta+\frac{\epsilon^2}{2}\nabla \log\pi(\theta),  \epsilon^2I_k)$ (Robert & Tweedie, 1996). In MALA, we use a Normal distribution whose mean is $\frac{\epsilon^2}{2}$​ away from the current value, $\theta$, in the direction of the first-order gradient of the target density, $\pi(\theta)$. In both Random Walk and MALA, we have the problem of finding the correct step size. A large step size may lead to an ineffective exploration of the space and low acceptance rate. A small step size can lead to a high acceptance rate but slow exploration of the space.

In NMC, we follow the same intuition and try to overcome the problem of choosing the step size by using the second-order gradient information. NMC offers different proposers based on the support of the random variable $\theta$, Real-Space proposer, Half-Space proposer and Simplex space proposer. One can also choose to transform the random variable $\theta$ from Half-Space or simplex into Real-Space and use the Real-Space proposer algorithm.

## Real-space NMC proposer

NMC uses a Multivariate Normal distribution to propose values in unconstrained space. Conceptually, NMC theorizes that the posterior is shaped like a Normal distribution. It uses first- and second-order gradient information from the current sample in order to build a proposal distribution whose shape matches that of this theoretical Normal distribution.

Mathematically, the proposal distribution $q$ used as a function of the current sample location $\theta$ is: $q(. \mid \theta)=\mathcal{N}(\theta-\nabla^2 \log\pi(\theta)^{-1}\nabla \log\pi(\theta), -\nabla ^2\log\pi(\theta)^{-1})$.

This proposal distribution works well the closer that the posterior is to a Normal distribution. However, in cases when the posterior is quite dissimilar to a Normal distribution, this proposer may overfit to its Normal assumption. To alleviate this, we introduce a "learning rate" $\gamma$, whose purpose is to reduce the strength of that Normal assumption when choosing a mean for the proposal distribution. The proposal distribution with the learning rate is: $q(. \mid \theta)=\mathcal{N}(\theta-\gamma\nabla^2 \log\pi(\theta)^{-1}\nabla \log\pi(\theta), -\nabla ^2\log\pi(\theta)^{-1})$. $\gamma\in[0, 1]$. $\gamma$ is initially drawn from $\text{Beta}(a, b)$, and an appropriate distribution for $\gamma$ is jointly learned with the rest of the model during inference ("Riemann manifold MALA", Section 5 of Girolami & Calderhead, 2011).

This proposer works for:

1. Random variables with unconstrained space support.
2. Random variables with constrained space support that are transformed into unconstrained space.

Subsequent sections will cover proposal distributions used in constrained spaces.

## Half-space NMC proposer

For random variables constrained with half-space support, NMC uses $\text{Gamma}$ as the base proposal distribution, and fits its parameters using local first- and second-order gradient information.

Mathematically, it is: $q(. \mid \theta)=\text{Gamma}(1-\theta^2\nabla^2\log\pi(\theta), -\theta\nabla^2\log\pi(\theta)-\nabla \log\pi(\theta))$.

## Simplex proposer

For random variables constrained with simplex-valued support, NMC uses the $\text{Dirichlet}$ distribution as the base proposal distribution, and fits its parameters using local first- and second-order gradient information.

For a $k$-dimensional simplex, the proposal distribution is $q(. \mid \theta)=\text{Dirichlet}(x;\alpha)$, where $\alpha \in \mathbb{R}^k$ and $\alpha_i = 1-\theta_i^2\left(\nabla_{ii}\log\pi(\theta)-\underset{i\neq j}{\max}\nabla_{ij}^2\log\pi(\theta)\right)$.

## NMC algorithm

Below is the NMC algorithm.

$\textbf{Input:} \text{ target density } \pi \text{ defined on } \theta_1...\theta_k $
$\textbf{Given: } a, b \text{ (default 1, 1)}$
$\textbf{repeat}$
$\quad \textbf{for } i=1\ \textbf{to}\ k\ \textbf{do}:$
$\quad\quad\text{Compute }\nabla \log\pi(\theta)\ \text{and}\ \nabla^2\log\pi(\theta)\ \text{w.r.t. }\theta_i$
$\quad\quad\text{Compute }q_i(.|\theta) \text{ as explained above}$
$\quad\quad\text{Sample }\theta^*\sim q_i(.|\theta)$
$\quad\quad\text{Set }\theta^*_j=\theta_j\text{ for }i\neq j$
$\quad\quad\text{Compute }\nabla \log\pi(\theta^*)\text{ and }\nabla^2\log\pi(\theta^*)\text{ w.r.t. }\theta_j^*$
$\quad\quad\text{Compute }q_i(.|\theta^*)$
$\quad\quad\alpha=\frac{\pi(\theta^*)q_i(\theta|\theta^*)}{\pi(\theta)q_i(\theta^*|\theta)}$
$\quad\quad u\sim \text{Uniform}(0,1)$
$\quad\quad\textbf{if }u<\alpha\textbf{ then}$
$\quad\quad\quad\theta=\theta^*$
$\quad\quad\textbf{end if}$
$\quad\textbf{end for}$
$\quad\text{Output sample }\theta$
$\textbf{until }\text{Desired number of samples}  $

Below is NMC algorithm with adaptive stage added to compute the learning rate parameters $a$ and $b$ for the $\text{Beta}$ distribution.

$\textbf{Input:} \text{ target density } \pi \text{ defined on } \theta_1...\theta_k $
$\textbf{Given: } a, b \text{ (default 1, 1)}$
$\textbf{repeat}$
$\quad \textbf{for } i=1\ \textbf{to}\ k\ \textbf{do}:$
$\quad\quad\text{Compute }\nabla \log\pi(\theta)\ \text{and}\ \nabla^2\log\pi(\theta)\ \text{w.r.t. }\theta_i$
$\quad\quad\textbf{if }\text{still in warm-up phase}\textbf{ then}$
$\qquad\qquad \text{Compute } a, b \text{ from mean and variance of the warmup samples using method of moments} $
$\qquad \textbf{end if}$
$\quad\quad\text{Sample }\gamma\sim \text{Beta}(a,b)$
$\quad\quad\text{Compute }q_i(.|\theta) \text{ as explained above}$
$\quad\quad\text{Sample }\theta^*\sim q_i(.|\theta)$
$\quad\quad\text{Set }\theta^*_j=\theta_j\text{ for }i\neq j$
$\quad\quad\text{Compute }\nabla \log\pi(\theta^*)\text{ and }\nabla^2\log\pi(\theta^*)\text{ w.r.t. }\theta_j^*$
$\quad\quad\text{Compute }q_i(.|\theta^*)$
$\quad\quad\alpha=\frac{\pi(\theta^*)q_i(\theta|\theta^*)}{\pi(\theta)q_i(\theta^*|\theta)}$
$\quad\quad u\sim \text{Uniform}(0,1)$
$\quad\quad\textbf{if }u<\alpha\textbf{ then}$
$\quad\quad\quad\theta=\theta^*$
$\quad\quad\textbf{end if}$
$\quad\textbf{end for}$
$\quad\text{Output sample }\theta$
$\textbf{until }\text{Desired number of samples}  $

## Using NMC

Here is how we call NMC, where $a$ and $b$ are `real_space_alpha` and `real_space_beta`.

```py
nmc = SingleSiteNewtonianMonteCarlo(real_space_alpha=1.0, real_space_beta=5.0)
samples = nmc.infer(
  queries=...,
  observations=...,
  num_samples=...,
  num_chains=...,
)
```

To enable adaptive stage run, we can set `num_adaptive_samples` during inference.

```py
nmc = SingleSiteNewtonianMonteCarlo(real_space_alpha=1.0, real_space_beta=5.0)
samples = nmc.infer(
  queries=...,
  observations=...,
  num_samples=...,
  num_chains=...,
  num_adaptive_samples=num_warmup_samples,
)
```

For random variables with half-space and simplex support, `SingleSiteNewtonianMonteCarlo` by default uses the half-space and simplex proposer. If a modeler is interested in transforming the random variables to real space and using the unconstrained proposer instead, we can write it as follows:

```py
nmc = SingleSiteNewtonianMonteCarlo(`transform_type=``TransformType``.``DEFAULT`)
samples = nmc.infer(
  queries=...,
  observations=...,
  num_samples=...,
  num_chains=...,
  num_adaptive_samples=num_warmup_samples,
)
```

---

Arora, N. S., Tehrani, N. K., Shah, K. D., Tingley, M., Li, Y. L., Torabi, N., Noursi, D., Masouleh, S. A., Lippert, E., and Meijer, E. (2020). Newtonian monte carlo: single-site mcmc meets second-order gradient methods. _arXiv:2001.05567_.

Metropolis, N., Rosenbluth, A., Rosenbluth, M., Teller, A., and Teller, E. (1953). “Equations of state calculations by fast computing machines.” _J. Chem. Phys._, 21(6): 1087–1092.

Robert, G., and Tweedie, R. 1996. Exponential convergence of Langevin diffusions and their discrete approximation. _Bernoulli_ 2:341–363.

Girolami, M., and Calderhead, B. (2011). “Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods: Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods.” Journal of the Royal Statistical Society: Series B (Statistical Methodology) 73, no. 2 (March 2011): 123–214. https://doi.org/10.1111/j.1467-9868.2010.00765.x.
