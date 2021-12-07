---
id: ancestral_metropolis_hastings
title: 'Single-Site Ancestral Metropolis-Hastings'
sidebar_label: 'Single-Site Ancestral MH'
slug: '/ancestral_metropolis_hastings'
---

Ancestral Metropolis-Hastings is one of the most fundamental Bayesian inference methods. In ancestral Metropolis-Hastings, values are sampled from the model's priors, and samples are accepted or rejected based on the sample's [Metropolis acceptance probability](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Formal_derivation). As such, ancestral Metropolis-Hastings is a very general inference method, making no strong assumptions about the structure of the model. However, this generality may lead it to be rather inefficient for many models.

Bean Machine provides a single-site variant of ancestral Metropolis-Hastings, in which values of random variables are sampled and updated one variable at a time (hence the name "single-site").

## Algorithm

Imagine we are using Single-Site Ancestral Metropolis-Hastings to choose a new value for a random variable $X$. Let's assume that $X$ is currently assigned a value $x$. Below are the steps to this algorithm:

1. First, we need to propose a new value for $X$. We do this by sampling from $X$'s prior distribution, and using that value as the new proposed value for $X$. Let's call that sampled value $x^\prime$.
2. Next, we need to identify other random variables that $X$ may have a _direct_ influence upon. This set of random variables is referred to as the [Markov blanket](https://en.wikipedia.org/wiki/Markov_blanket) of $X$. The Markov blanket of a random variable consists of the random variable's parents (those that it depends upon), children (those that depend upon it), and the other parents of the random variable's children. We only need to consider the Markov blanket of random variable $X$ when assessing appropriateness, because only the likelihoods of these distributions are directly affected by a change in the value of $X$. All other random variables in the model are conditionally independent of $X$ given the random variables in $X$'s Markov blanket.
3. Now, we need to assess whether this sample is appropriate. We will examine the likelihood of $x^\prime$, conditional on the other variables in its Markov blanket. We can do this computationally by computing the (log) likelihoods of those other random variables when $X = x^\prime$.
4. Finally, we compare the (log) likelihoods when $X = x$ and $X = x^\prime$. We use the [Metropolis acceptance probability](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Formal_derivation) to accept $x^\prime$ that tend to have relatively higher (log) likelihoods. The exact acceptance probability can be read about in the linked article, or in the algorithm details below.

## Details

This is the standard ancestral Metropolis-Hastings algorithm:

$\textbf{Input: }\text{evidence }\mathcal{E} \text{ and queries } \mathcal{R}\\$
$\textbf{Given: }\text{a family of proposal distributions }\mathcal{Q}\\$
$\textbf{Create: } \text{initial world }\sigma\text{ initialized with }\mathcal{E}\text{ and extended to include }\mathcal{R}\\$
$\textbf{repeat}\\$
$\qquad\text{Let }V\text{ represent random variables in }\sigma\text{ excluding }\mathcal{E}\\$
$\qquad\textbf{for }X\textbf{ in }V\textbf{do}\\$
$\qquad\qquad\text{Sample }x'\sim\mathcal{Q}_X(. \mid \sigma)\\$
$\qquad\qquad\text{Clone }\sigma\text{ to }\sigma'\text{ and set }\sigma'_X=x'\\$
$\qquad\qquad\text{Recompute }\sigma'_{Y}\text{ for }Y\in\text{ children of } X\text{ in }\sigma'\\$
$\qquad\qquad\alpha=\min\left[1, \frac{p(\sigma')\mathcal{Q}_X(\sigma_X\mid\sigma')}{p(\sigma)\mathcal{Q}_X(\sigma'_X\mid\sigma)}\right]\\$
$\qquad\qquad u\sim \text{Uniform(0, 1)}\\$
$\qquad\qquad\textbf{if }u<\alpha\textbf{ then}\\$
$\qquad\qquad\qquad\sigma=\sigma'\\$
$\qquad\qquad\textbf{end if}\\$
$\qquad\textbf{end for}\\$
$\qquad\text{Emit sample }\sigma\\$
$\textbf{until }\text{Desired number of samples}$

Or, in pseudo-code:

```
For each inference iteration:
    For each unobserved random variable X:
        Perform a Metropolis Hastings (MH) update, which involves:
            1. Propose a new value x′ for X using proposal Q
            2. Update the world σ to σ′
            3. Accept / reject the new value x' using Metropolis acceptance probability
```

## Usage

```py
samples = bm.SingleSiteAncestralMetropolisHastings().infer(
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
