---
id: beanstalk
title: The Beanstalk Compiler
sidebar_label: 'Beanstalk'
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
- Inference algorith - currently, only Newtonian Monte Carlo (NMC), is supported,
- Only one chain of samples can be generated at a time.

To use Beanstalk to run an inference model, intsead of using a standard Bean Machine inference algorithm using a command such as `bm.SingleSiteNewtonianMonteCarlo().infer`, simply include the compiler using `from beanmachine.ppl.inference.bmg_inference import BMGInference` and use `BMGInference().infer`.

The `BMGInference()` object provides three utility methods that can be used to inspect the intermediate results of the compiler, namely:
- `BMGInference().to_dot(queries, observations)` - Returns a DOT representation of the probabilistic graph of the model,
- `BMGInference().to_cpp(queries, observations)` - Returns a C++ program that builds a version of this graph, and
- `BMGInference().to_python(queries, observations)` - Returns a Python program that builds a version of the graph.
