---
id: toc
title: Table of Contents
sidebar_label: TOC
slug: /users
---
* [Landing page](landing_page/landing_page.md)
* [Bean Machine advantages](landing_page/bean_machine_advantages.md)

**Overview**

* [Introduction](overview/introduction/introduction.md)
* [Quick start](overview/quick_start/quick_start.md)
* [Modeling](overview/modeling/modeling.md)
* [Inference](overview/inference/inference.md)
* [Post-inference](overview/analysis/analysis.md)

**Framework topics**

* **Declarative syntax**
  * Declarative philosophy
  * `@random_variable`
  * Model composition and namespacing
  * Worlds
* **[Inference](framework_topics/inference/inference.md)**
  * Inference methods
    * [Ancestral Metropolis-Hastings](framework_topics/inference/ancestral_metropolis_hastings.md)
    * [Uniform sampler](framework_topics/inference/uniform_metropolis_hastings.md)
    * [Random Walk Metropolis-Hastings](framework_topics/inference/random_walk.md)
    * Metropolis-Adjusted Langevin Algorithm
    * [Hamiltonian Monte Carlo and NUTS](framework_topics/inference/hamiltonian_monte_carlo.md)
    * [Newtonian Monte Carlo](framework_topics/inference/newtonian_monte_carlo.md)
* **Programmable inference**
  * [Programmable inference](framework_topics/programmable_inference/programmable_inference.md)
  * [Compositional inference](framework_topics/programmable_inference/compositional_inference.md)
  * [Block inference](framework_topics/programmable_inference/block_inference.md)
  * [Transformations](framework_topics/programmable_inference/transforms.md)
  * [Warmup and adaptation](framework_topics/programmable_inference/adaptive_inference.md)
* **Custom proposers**
  * [Custom proposers](framework_topics/custom_proposers/custom_proposers.md)
  * Mixture of proposers
  * [Variable API](framework_topics/custom_proposers/variable.md)
  * Diff and diffstack
* **Model evaluation**
  * [Diagnostics](framework_topics/model_evaluation/diagnostics.md)
  * [Model comparison](framework_topics/model_evaluation/model_comparison.md)
  * [Posterior predictive checks](framework_topics/model_evaluation/posterior_predictive_checks.md)
  * Generating data <!-- simulate should go here! -->
  * Custom diagnostics <!-- optional for now -->
* **Development**
  * Debugging tips
  * Modeling tips
  * [Logging](framework_topics/development/logging.md)

**Advanced topics**
<!-- I don't think we'll plan to have any of these ready for a while. -->

* Gaussian processes
* Bean Machine Graph
* Beanstalk

**API**
<!-- Brian Johnson will link this in. -->

**Tutorials**

* [Coin flipping](https://www.internalfb.com/intern/anp/view/?id=277521)
* [Linear regression](https://www.internalfb.com/intern/anp/view/?id=282519)
* [Logistic regression](https://www.internalfb.com/intern/anp/view/?id=280068)
* [Sparse logistic regression](https://www.internalfb.com/intern/anp/view/?id=275391)
* [Gaussian Mixture Model](https://www.internalfb.com/intern/anp/view/?id=270772)
* [Hidden Markov Model](https://www.internalfb.com/intern/anp/view/?id=273851)
* [Neal's Funnel](https://www.internalfb.com/intern/anp/view/?id=273308)
* Robust regression
* $n$-schools

* **Advanced**
  * [Dynamic bistable hidden Markov model](https://www.internalfb.com/intern/anp/view/?id=275944)
