---
id: inference
title: 'Inference Methods'
sidebar_label: 'Inference Methods'
slug: '/inference'
---
<!-- @import "../../header.md" -->

Posterior distributions can often only be estimated, as the solution such problems in general have no closed-form. Bean Machine's inference methods include sequential sampling techniques known as [Markov chain Monte Carlo (MCMC)](https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50) to generate samples representative of this distribution. These posterior distribution samples are the main output of Bean Machine: with enough samples, they will reliably resemble the posterior distribution for the problem at hand.

To support inference algorithms, Bean Machine represents the model as a directed acyclic graph where each node is a random variable and edges between nodes represent dependencies between random variables. During a single iteration of inference, MCMC assigns a specific, concrete value to each of the unobserved random variable functions in your model. We refer to this assignment as a "world".

Each world corresponds to a potential sample for the posterior distribution. An MCMC method evaluates how well a particular world would explain the observed data (and prior beliefs). MCMC methods will tend to retain worlds that explain the observed data well and add them as samples to the computed posterior distribution. MCMC methods will tend to discard worlds that do a poor job of explaining the observed data.

In an MCMC method, worlds are computed sequentially. A new world is "proposed" based on the random variable assignments from the current world. In each inference step, an MCMC method iterates over all unobserved random variables and proposes a new value. The world is updated to reflect this change; that is, likelihoods are updated and new variables may be added or removed. This updated world will either replace the existing world or be discarded as determined by the specific inference method. The value associated with each variable at the $i$th inference step is returned as the $i$th sample for the variable.

As you can imagine, there are a variety of ways of proposing new worlds from the current world, and even for deciding whether to accept or reject a proposed world. Lots of research goes into designing inference methods that are both flexible and performant for a wide class of models. Bean Machine supports many inference methods out-of-the-box, and which are described in the rest of this section of the documentation. This section also covers a particularly promising feature of Bean Machine, namely, programmable inference.
