---
id: beanstalk
title: The Beanstalk Compiler
sidebar_label: 'Compiler'
---

<!-- @import "../../header.md" -->

### What is Beanstalk?

Beanstalk is an experimental compiler under active development: it transforms models written in Bean Machine into an optimized [Bean Machine Graph (BMG) C++ Runtime](../bmg/bmg.md) model.
It is designed specifically to handle "unvectorized" models where stochastic quantities are
tensors which contain exactly one value; we can often obtain significant performance improvements
over Bean Machine  when using BMG to run inference on such models.

The tutorials currently working with Beanstalk are:
- Linear regression
- Gaussian mixture model
- Neal's funnel

For the above three models, the Beanstalk-compiled version of NMC inference reduces runtime to generate samples of size 10K for the posterior distribution by anywhere between 80x and 250x depending on the model.

### Model restrictions

Models compiled with Beanstalk have many restrictions. In the current release:

- With some exceptions, all tensor quantities manipulated by the model must be single-valued. There is
  some limited support for one- and two-dimensional tensors.
- `@random_variable` functions must return a univariate `Bernoulli`, `Beta`, `Binomial`, `Categorical`,
  `Chi2`, `Dirichlet`, `Gamma`, `HalfCauchy`, `HalfNormal`, `Normal`, `StudentT` or `Uniform(0., 1.)`
  distribution.
- Tensor operators on stochastic values are limited to `add()`, `div()`, `exp()`, `expm1()`,
  `item()`, `log()`, `logsumexp()`, `mul()`, `neg()`, `pow()`, `sigmoid()` and `sub()`.
- Python operators on stochastic values in `@random_variable` or `@functional` functions are limited to
  `+`, `-`, `*`, `/`, and `**` operators. Matrix multiplication and comparisons are not yet supported.
- Support for the `[]` indexing operation is limited.
- Support for "destructuring" assignments such as `x, y = z` where `z` is a stochastic quantity is limited.
- All `@random_variable` and `@functional` functions in the model *and every function called by them*
  must be "pure". That is, the value returned must be logically identical every time the function is
  called with the same arguments, and the function must not modify any externally-observable state.
- Models must not mutate existing tensors "in place"; always create new values rather than mutating
  existing tensors.
- Conditions of `while` statements, `if` statements, and `if` expressions must not be stochastic.

### Getting started with Beanstalk

To use Beanstalk to run Bean Machine Graph inference on a Bean Machine model, first import the inference engine. For example: `from beanmachine.ppl.inference.bmg_inference import BMGInference`.

The `BMGInference()` object provides the following methods to inspect the compiler's analysis and produce code that generates a Bean Machine Graph model:

- `BMGInference().infer(queries, observations, num_samples, num_chains)` - Compiles the model and executes
  inference using Bean Machine Graph; returns a dictionary of samples for the queried variables. In the current
  release only Newtonian Monte Carlo (NMC) is supported when running inference with `BMGInference`.

- `BMGInference().to_cpp(queries, observations)` - Returns a C++ program fragment that constructs the equivalent
BMG model.
- `BMGInference().to_python(queries, observations)` - Returns a Python program that constructs the equivalent
BMG model.
- `BMGInference().to_graphviz(queries, observations)` - Returns a graphviz graph representing the model.
- `BMGInference().to_dot(queries, observations)` - Returns the DOT source code of the graphviz graph.


<FbInternalOnly>

Facebook specific:

 These models are also frequently used at Facebook including Team Power and Metric Ranking products (https://fb.workplace.com/notes/418250526036381) as well as new pilot studies on https://fb.quip.com/GxwQAIscFRz8 and https://fb.quip.com/UMmcAr2zczbc. Additionally, the Probabilistic Programming Languages (https://www.internalfb.com/intern/bunny/?q=group%20pplxfn) (PPL) team has collected a list of https://fb.quip.com/rrMAAuk02Jqa who can benefit from our HME methodology.

BMG: https://fb.quip.com/TDA7AIjRmScW

</FbInternalOnly>
