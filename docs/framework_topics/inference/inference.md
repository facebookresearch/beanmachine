<!-- @import "../../header.md" -->

# Inference

Posterior distributions can often only be estimated, as the solution such problems in general have no closed-form. Bean Machine's inference methods typically use sequential sampling techniques known as [Markov chain Monte Carlo (MCMC)](https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50) to generate samples representative of this distribution. These posterior samples are the main output of Bean Machine: with enough samples, they will reliably resemble the posterior distribution for the problem at hand.

To support inference algorithms, Bean Machine represents the model as a directed acyclic graph where each node is a random variable and edges between nodes represent dependencies between random variables. During a single iteration of inference, Bean Machine assigns a specific, concrete value to each of the unobserved random variable functions in your model. We refer to this assignment as one "world" in Bean Machine.

Each world corresponds to a potential sample for the posterior distribution. Bean Machine evaluates how well a particular world would explain the observed data (and prior beliefs). Bean Machine will tend to retain worlds that explain the observed data well and add them as samples to the computed posterior distribution. Bean Machine will tend to discard worlds that do a poor job of explaining the observed data.

As a part of MCMC inference, worlds are computed sequentially. A new world is "proposed" based on the random variable assignments from the current world. In each inference step, Bean Machine iterates over all unobserved random variables and proposes a new value. The world is updated to reflect this change; that is, likelihoods are updated and new variables may be added or removed. This updated world will either replace the existing world or be discarded as determined by the specific inference method. The value associated with each variable at the $i$th inference step is returned as the $i$th sample for the variable.

## Inference methods

As you can tell, there are a variety of ways of proposing new worlds from the current world, and even for deciding whether to accept or reject a proposed world! This process is collectively referred to as an inference method. Lots of research goes into designing inference methods that are both flexible and performant for a wide class of models. Bean Machine supports many inference methods out-of-the-box. Check out the docs under **Inference methods** to learn more!

## Programmable inference

One of the key innovations in Bean Machine is the idea that inference is programmable. Bean Machine's single-site paradigm allow you to modularly mix-and-match inference components to get the most out of your model.

Compositional inference allows you to utilize distinct inference methods for different random variables when fitting a model. Our flexible transformations framework allows you to leverage domain-specific transformations or proposers, which can be especially powerful to avoid worse edge-case performance when running inference over constrained random variables. Block inference allows you to propose updates for several random variables jointly, which can be necessary when dealing with highly-correlated variables. All of these techniques are covered in the **Programmable inference** section.

We've also included a section on Bean Machine's framework for implementing and manipulating **Custom proposers**. This framework exposes tools for building domain-specific proposers which can be plugged in modularly with many of Bean Machine's existing inference methods.
