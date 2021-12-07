---
id: bean_machine_advantages
title: Bean Machine Advantages
sidebar_label: Bean Machine Advantages
---

:::tip

If you're new to probabilistic programming languages, we recommend you skip to the [next page](../quick_start/quick_start.md)!

:::

By building on top of PyTorch with a declarative modeling syntax, Bean Machine can be simultaneously performant and intuitive for building probabilistic models. Bean Machine provides further value by implementing cutting-edge inference algorithms and allowing the user to select and program custom inferences for different problems and subproblems.

## Site-based inference

<!-- ### Single-site inference -->

Bean Machine uses a site-based inference engine. "Sites" are random variable families, and Bean Machine uses these families to enable a modular inference engine.

The simplest form of site-based inference is called "single-site" inference. In the single-site paradigm, models are built up from random variables that can be reasoned about individually. Bean Machine can exploit this modularity to update random variables one-at-a-time, reducing unnecessary computation and enabling posterior updates that might not be possible if processing the entire model in one go.

Bean Machine also supports "multi-site" inference, in which multiple sites are reasoned about jointly. This increases complexity during inference, but it allows the engine to exploit inter-site correlations when fitting the posterior distribution.

Altogether, site-based inference is a flexible pattern for trading off complexity and modularity, and enables the advanced techniques outlined below.

## Declarative modeling

<!-- Usability improvements -->
In Bean Machine, random variables are implemented as decorated Python functions, which naturally form an interface for the model. Using functions makes it simple to determine a random variable's definition, since it is contained in a function that is usually only a few lines long. This lets you directly refer to random variables to access inferred distributions or when binding data to your model. This is safer and more natural than relying on string identifiers, and also enables IDE support and type-checking in many cases.

<!-- Efficiency improvements -->
Declarative modeling also frees the inference engine to reorder model execution. Foremost, it enables computation of immediate dependencies for random variables. This makes it possible to propose new values for a random variable by examining only its dependencies, saving significant amounts of compute in models with complex structure.

## Programmable inference

<!-- Compositional inference -->
Bean Machine allows the user to design and apply powerful inference methods. Because Bean Machine can propose updates for random variables individually, the user is free to customize the _method_ which it uses to propose those values. Different inference methods can be supplied for different families of random variables. For example, a particular model can leverage gradient information when proposing values for differentiable random variables, and at the same time might sample from discrete ones with a particle filter. This single-site "compositional inference" pattern enables seamless interoperation among any MCMC-based inference strategies.

<!-- Block inference -->
Though powerful, compositional inference limits Bean Machine's global understanding of the model. To combat this, Bean Machine exposes a separate functionality to allow joint processing of multiple sites. This "multi-site inference" causes Bean Machine to process both sites together before updating either, which is especially useful for updating highly-correlated random variables. Certain inference methods may be able to further exploit multi-sites with inference-specific optimizations.  Since multi-site inference is orthogonal to compositional inference, it allows you to create sophisticated, model-specific inference strategies with virtually no additional effort.

## Advanced methods

Bean Machine supports a variety of classic inference methods such as ancestral sampling and the No-U-Turn sampler (NUTS). However, the framework also leverages single-site understanding of the model in order to provide efficient methods that take advantage of higher-order gradients and model structure.

<!-- NMC -->
Bean Machine includes the first implementation of Newtonian Monte Carlo (NMC) in a more general platform. NMC utilizes second-order gradient information to construct a multivariate Gaussian proposer that takes local curvature into account. As such, it can produce sample very efficiently with no warmup period when the posterior is roughly Gaussian. Bean Machine's structural understanding of the model lets us keep computation relatively cheap by only modeling a subset of the space that is relevant to updating a particular random variable.

<!-- Custom proposers -->
For certain domains, prepackaged inference methods may not be the best tool for the job. For example, if dealing with a problem specified in spherical coordinates, it may be useful to incorporate a notion of spherical locality into the inference proposal. Or, you may want to incorporate some notion of ordering when dealing with certain discrete random variables. Bean Machine exposes a flexible abstraction called _custom proposers_ for just this problem. Custom proposers let the user design powerful new inference methods from the building blocks of existing ones, while easily plugging into Bean Machine's multi-site paradigm.

## Bean Machine Graph compilation

<!-- PyTorch performance -->
PyTorch offers strong performance for models comprised of a small number of large tensors. However, many probabilistic models have a rich or sparse structure that is difficult to write in terms of just a handful of large tensor operations. And in many cases, these are exactly the problems for which probabilistic modeling is most compelling!

To address this, we are developing an experimental inference runtime called Bean Machine Graph (BMG) Inference. BMG Inference is a specialized combination of compiler and a fast, independent runtime that is optimized to run inference even for un-tensorized models. By design, BMG Inference has the same interface as other Bean Machine inference methods, relying on a custom behind-the-scenes compiler to interpret your model and translate it to a faster implementation with no Python dependencies.

BMG Inference routinely achieves 1 to 2 orders-of-magnitude speedup for untensorized models. However, please note that this infrastructure is under development, and the supported feature set may be limited.
