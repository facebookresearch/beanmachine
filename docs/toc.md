---
slug: toc
title: Sitemap
sidebar_label: Sitemap
---
* [Landing page](landing_page/landing_page.md) [DOES NOT GO TO LANDING PAGE]

**Overview**

* [Introduction](overview/introduction/introduction.md)
* [Bean Machine advantages](overview/bean_machine_advantages/bean_machine_advantages.md)
* [Quick start](overview/quick_start/quick_start.md)
* [Modeling](overview/modeling/modeling.md)
* [Inference](overview/inference/inference.md)
* [Analysis](overview/analysis/analysis.md)
* [Installation](overview/installation/installation.md)

**Framework topics**

* **Declarative Syntax**
  * Declarative Philosophy
  * `@random_variable`
  * Model Composition and Namespacing
  * Worlds
* **[Inference](framework_topics/inference/inference.md)**
  * **Inference methods**
    * [Uniform Metropolis-Hastings](framework_topics/inference/uniform_metropolis_hastings.md)
    * [Random Walk Metropolis-Hastings](framework_topics/inference/random_walk.md)
    * [Ancestral Metropolis-Hastings](framework_topics/inference/ancestral_metropolis_hastings.md)
    * Metropolis-Adjusted Langevin Algorithm
    * [Hamiltonian Monte Carlo and NUTS](framework_topics/inference/hamiltonian_monte_carlo.md)
    * [Newtonian Monte Carlo](framework_topics/inference/newtonian_monte_carlo.md)
  * **[Programmable Inference](framework_topics/programmable_inference/programmable_inference.md)**
    * [Transformations](framework_topics/programmable_inference/transforms.md)
    * [Block Inference](framework_topics/programmable_inference/block_inference.md)
    * [Compositional Inference](framework_topics/programmable_inference/compositional_inference.md)
    * [Adaptation and Warmup](framework_topics/programmable_inference/adaptive_inference.md)
  * **Custom Proposers**
    * [Worlds and Variables API](framework_topics/custom_proposers/variable.md)
    * [Proposers API](framework_topics/custom_proposers/custom_proposers.md)
    * Mixture of proposers
    * Diff and diffstack
* **Model evaluation**
  * [Diagnostics Module](framework_topics/model_evaluation/diagnostics.md)
  * [Posterior Predictive Checks](framework_topics/model_evaluation/posterior_predictive_checks.md)
  * [Model Comparison](framework_topics/model_evaluation/model_comparison.md)
  * Generating data <!-- simulate should go here! -->
  * Custom diagnostics <!-- optional for now -->
* **Advanced topics**
<!-- I don't think we'll plan to have any of these ready for a while. -->
  * Gaussian processes
  * [Bean Machine Graph Inference](overview/beanstalk/beanstalk.md)
* **Development**
  * Debugging tips
  * Modeling tips
  * [Logging](framework_topics/development/logging.md)


**API**
<!-- Brian Johnson will link this in. See ../website/sidebars.js for where we think 'API' will go... -->

**[Packages](overview/packages/packages.md)**

**[Tutorials](overview/tutorials/tutorials.md)**

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
