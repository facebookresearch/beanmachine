---
id: overview
title: Welcome
sidebar_label: Welcome
---


Bean Machine is a new universal probabilistic programming language to enable fast and accurate Bayesian analysis. Today, we’re using Bean Machine to platformize Bayesian analysis across Facebook. Bean Machine is built atop PyTorch, and touts a modeling engine that exploits state-of-the-art gradient-based inference techniques to scale to large datasets. Its declarative modeling language allows for the expression of deep, domain-specific models, and can deliver directly-interpretable results that are robust-by-default. We have battle-tested Bean Machine across critical domains in Facebook including Integrity, Experimentation, and Networking to demonstrate its cutting-edge performance, and we are excited to open source the platform later this year!

## Why Bayesian?

Methods like neural networks offer excellent insights on **massive datasets** where the underlying **model is unknown**. Contrastingly, Bayesian methods excel on **small and medium datasets** where the underlying **model is interpretable**. Concretely, Bayesian analysis offers 3 major benefits:

1. **Uncertainty.**  Predictions are quantified with reliable measures of uncertainty (probability distributions). An analyst can understand not only the system’s prediction, but how confident the system is about it.
2. **Expressivity.**  Bayesian modeling allows a developer to write out a causal or graphical probabilistic model directly in source code. This allows them to match the structure of their model to the structure of problem they’re trying to solve.
3. **Interpretability.**  Because the model matches the domain, developers can query intermediate properties (latent variables) within the model. This can be used to interpret *why* a particular prediction was made, and can aid the model development process.

You can find more documentation on Bean Machine on the left. You can check out source code documetation under API above, and you can you can find interactive examples in the Tutorials section.
