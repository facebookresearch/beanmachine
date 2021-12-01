---
id: ancestral_metropolis_hastings
title: 'Single-site Ancestral Metropolis-Hastings'
sidebar_label: 'Single-site Ancestral Metropolis-Hastings'
slug: '/ancestral_metropolis_hastings'
---

Ancestral Metropolis-Hastings is one of the most fundamental Bayesian inference methods. In ancestral Metropolis-Hastings, values are sampled from the model's priors, and samples are accepted or rejected based on the sample's Metropolis acceptance probability. As such, ancestral Metropolis-Hastings is a very general inference method, making no strong assumptions about the structure of your model; however, this generality may lead it to be rather inefficient for many models.

Bean Machine provides a single-site variant of ancestral Metropolis-Hastings, in which values of random variables are sampled and updated one variable at a time (hence the name "Single-Site").

There are four main steps in Single Site Ancestral Metropolis-Hastings.


1. Propose a value for a random variable, which we'll call $X$. The proposed value is sampled from the random variable's prior distribution.
2. Given the proposed value for $X$, update the distributions of all random variables in the Markov blanket of $X$. The Markov blanket of a random variable consists of the random variable's children (those that depend upon it), parents (those that it depends upon), and the other parents of the random variable's children. We only consider the Markov blanket of random variable $X$ because only the likelihoods of these distributions are directly affected by a change in the value of $X$. All other random variables in the model are conditionally independent of $X$ given the random variables in $X$'s Markov blanket.
3. Compute the proposal ratio of proposing this new value given the updated distributions of all random variables in the Markov blanket of random variable $X$. This is known as the Metropolis-Hastings acceptance ratio.
4. Accept or reject the proposed value with the probability computed in Step 3.

## Algorithm

This is the standard ancestral Metropolis-Hastings algorithm:

$\textbf{Input: }\text{evidence }\mathcal{E} \text{ and queries } \mathcal{R}$
$\textbf{Given: }\text{a family of proposal distributions }\mathcal{Q}$
$\textbf{Create: } \text{initial world }\sigma\text{ initialized with }\mathcal{E}\text{ and extended to include }\mathcal{R}$
$\textbf{repeat}$
$\qquad\text{Let }V\text{ represent random variables in }\sigma\text{ excluding }\mathcal{E}$
$\qquad\textbf{for }X\textbf{ in }V\textbf{do}$
$\qquad\qquad\text{Sample }x'\sim\mathcal{Q}_X(. \mid \sigma)$
$\qquad\qquad\text{Clone }\sigma\text{ to }\sigma'\text{ and set }\sigma'_X=x'$
$\qquad\qquad\text{Recompute }\sigma'_{Y}\text{ for }Y\in\text{ children of } X\text{ in }\sigma'$
$\qquad\qquad\alpha=\min\left[1, \frac{p(\sigma')\mathcal{Q}_X(\sigma_X\mid\sigma')}{p(\sigma)\mathcal{Q}_X(\sigma'_X\mid\sigma)}\right]$
$\qquad\qquad u\sim \text{Uniform(0, 1)}$
$\qquad\qquad\textbf{if }u<\alpha\textbf{ then}$
$\qquad\qquad\qquad\sigma=\sigma'$
$\qquad\qquad\textbf{end if}$
$\qquad\textbf{end for}$
$\qquad\text{Emit sample }\sigma$
$\textbf{until }\text{Desired number of samples}$

Or, in pseudo-code:

```
For each inference iteration:
    For each unobserved random variable X:
        Perform a Metropolis Hastings (MH) update, which involves:
            1. Propose a new value x′ for X using proposal Q
            2. Update the world σ to σ′
            3. Accept / reject the new value x' using Metropolis-Hastings acceptance ratio
```

## Usage

Here is an example of how to use Single-Site Ancestral Metropolis Hastings to perform inference in Bean Machine.

```py
from beanmachine.ppl.inference.single_site_ancestral_mh import SingleSiteAncestralMetropolisHastings

mh = SingleSiteAncestralMetropolisHastings()
coin_samples = mh.infer(queries, observations, num_samples, num_chains)
```

The parameters to `infer` are described below:

| Name | Usage
| --- | ---
| `queries` | A list of `@bm.random_variable` targets to fit posterior distributions for.
| `observations` | The `Dict` of observations, where each key is a random variable, and its value is the observed value for that random variable.
| `num_samples` | Number of samples to build up distributions for the values listed in `queries`.
| `num_chains` | Number of separate inference runs to use. Multiple chains can verify inference ran correctly.
