---
id: beanstalk
title: The Beanstalk Compiler
sidebar_label: 'Compiler'
---

<!-- @import "../../header.md" -->

_This page is Work in Progress!_

Beanstalk is an experimental, just-in-time (JIT) compiler for Bean Machine. While we expect to continue to develop this compiler in the near future, currently it handles only a subset of the Bean Machine language. For example, it supports the following tutorials:
- Linear Regression,
- Gaussian Mixture Model (1D, mixture of 2) - TODO: The currently included tutorial is more general than that,
- Neal's funnel.

The subset currently is limited to:
- Univariate distributions,
- Simple uses of tensors, for example, tensor addition and multiplication,
- Limited control flow is supported,
- Inference algorithms - currently only Newtonian Monte Carlo (NMC) is supported,
- Only one chain of samples can be generated at a time.

To use Beanstalk to run an inference model, instead of using a standard Bean Machine inference algorithm using a command such as `bm.SingleSiteNewtonianMonteCarlo().infer()`, simply include the compiler using `from beanmachine.ppl.inference.bmg_inference import BMGInference` and use `BMGInference().infer()`.

The `BMGInference()` object provides a collection of utility methods that can be used to inspect the intermediate results of the compiler, namely:
- `BMGInference().infer(queries, observations, num_samples, num_chains)` - Returns a dictionary of samples for the queried variables,
- `BMGInference().to_graphviz(queries, observations)` - Returns a graphviz graph representing the model,
- `BMGInference().to_dot(queries, observations)` - Returns a DOT representation of the probabilistic graph of the model,
- `BMGInference().to_cpp(queries, observations)` - Returns a C++ program that builds a version of this graph, and
- `BMGInference().to_python(queries, observations)` - Returns a Python program that builds a version of the graph.

### Beanstalk uses the Bean Machine Graph (BMG) library
With code generated that is powered by the Bean Machine Graph (BMG) library, which runs critical pieces of code in C++ rather than Python, to speed up the inference process significantly.

-----------

Facebook specific:

 These models are also frequently used at Facebook including Team Power and Metric Ranking products (https://fb.workplace.com/notes/418250526036381) as well as new pilot studies on https://fb.quip.com/GxwQAIscFRz8 and https://fb.quip.com/UMmcAr2zczbc. Additionally, the Probabilistic Programming Languages (https://www.internalfb.com/intern/bunny/?q=group%20pplxfn) (PPL) team has collected a list of https://fb.quip.com/rrMAAuk02Jqa who can benefit from our HME methodology.

BMG: https://fb.quip.com/TDA7AIjRmScW

Ignore--saved for formatting tips:
Let's quickly translate the model we discussed in the [Introduction](../introduction/introduction.md) into Bean Machine code! Although this will get you up-and-running, **it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine**. Happy modeling!
