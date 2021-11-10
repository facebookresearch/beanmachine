---
id: introduction
title: 'Introduction'
sidebar_label: 'Introduction'
slug: '/introduction'
---
<!-- @import "../../header.md" -->

Bean Machine is a probabilistic programming language that makes developing and deploying generative probabilistic models intuitive and efficient.

## Why generative models?

Bean Machine's generative modeling is concerned not only with providing useful predictions (as traditional ML techniques do), but also with estimating the uncertainty inherent in the problem at hand in the form of probability distributions. Estimating uncertainty helps ensure that predictions are reliable and robust.

Generative modeling with Bean Machine offers many benefits:

  1. **Uncertainty estimation.**  Predictions are quantified with reliable measures of uncertainty in the form of probability distributions. An analyst can understand not only the system's prediction, but also the relative likelihood of other possibilities.
  2. **Expressivity.**  It's extremely easy to encode a rich model directly in source code. This allows one to match the structure of the model to the structure of the problem.
  3. **Interpretability.**  Because the model matches the domain, one can query intermediate learned properties within the model. This can be used to interpret *why* a particular prediction was made, and can aid the model development process.

## Generative probabilistic models

A generative probabilistic model consists of **random variables** and **conditional probability distributions** (CPD) that encode knowledge about some domain. For example, consider a simplified model for the spread of infectious diseases where we wish to express the fact that the number of new cases on a given day is on average the current number of infected persons times the daily reproduction rate of the disease. In order to mathematically state this probabilistic fact, it is a common practice to rely on _elementary probability distributions_ (EPD) with well known statistics such as _Poisson_ as follows:


```
num_new_cases ~ Poisson(reproduction_rate * num_infected)
```

![](/img/poisson_3.png)

<!-- It might be more interesting to show the Poisson distribution for reproduction_rate=0.1 and num_inference=1000000 -->


Let's assume that `num_infected` is a constant, then the above statement gives the CPD of the random variable `num_new_cases` conditioned on the value of its **parent random variable**, `reproduction_rate`.
Since the parameter of the Poisson distribution is also its mean, this CPD is consistent with the knowledge that we were trying to express.

A well-formed generative model must specify the CPD of each random variable and the graph induced by the parent-child relationship between random variables must be acyclic. To complete this model, we must therefore specify the CPD of the random variable `reproduction_rate`. In the case of new diseases where we don't know anything about the actual reproduction rate, this poses a seemingly intractable problem. The **Bayesian approach** suggests a solution to this problem. In this approach, we specify our *a-priori* belief (i.e. our belief before seeing any data) about the distribution of random variables as their CPD. So, in this example, if infectious disease experts believe that a new disease would have a daily reproduction rate drawn from a distribution with a mean of 0.1, we could express this belief with the help of another EPD, the *Exponential* distribution, as follows:

```
reproduction_rate ~ Exponential(1 / 0.1)
```

![](/img/exponential_10.png)

## Inference

Given a generative model, the natural next step is to use it to perform inference. Inference is the process of combining a _model_ with _data_ to obtain _insights_, in the form of probability distributions over values of interest. Our documentation refers to the data as "observations", the insights as "posterior distributions", and the values of interest as "queried random variables".

In our example above, let's say we observe that `num_infected` is 1087980 and that `num_new_cases` is 238154. Now, given this observation, we might want to query the posterior distribution for `reproduction_rate`. Or, mathematically speaking, we seek the following,

$$\mathbb{P}(\texttt{reproduction\_rate} \mid \texttt{num\_new\_cases} = 238154, \texttt{num\_infected}=1087980)$$

One way to understand the semantics of the inference task is to think of a generative probabilistic model as specifying a distribution over possible **worlds**. A world can be thought of as an assignment of values to all random variables in the model. So, for example, some possible worlds are:

- `reproduction_rate` = 0.01, `num_new_cases` = 9000,
- `reproduction_rate` = 0.1, `num_new_cases` = 90000, or
- `reproduction_rate` = 0.9, `num_new_cases` = 800000.

Our generative model specifies a probability distribution over each of these worlds. Now, the inference task is to restrict attention to only worlds where `num_new_cases` = 238154. We're interested in learning `reproduction_rate` within these worlds.

## Where does Bean Machine fit in?

In the rest of this Overview we'll introduce you to Bean Machine's syntax, and show how it can be used to learn about problems like this one. Traditionally, lots of painstaking, hand-crafted work has gone into modeling generative scenarios. Bean Machine aims to handle all of the manual work involved in fitting data to your model, leaving you to focus on the exciting part: the problem itself! Keep on reading to find out how.

## Target audience

While we hope that the guides you'll find here are relevant to anyone with an ML background, there are excellent resources available if this is your first exposure to Bayesian analysis! We highly recommend the excellent YouTube series _[Statistical Rethinking](https://www.youtube.com/playlist?list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI)_, which walks through Bayesian thinking and probabilistic modeling. For a more hands-on experience, you can check out the free, online tutorial _[Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/#contents)_.

<!-- ## Sneak Peek at Bean Machine Syntax

At this point it would be helpful to briefly look at our example model written in Bean Machine.
Random variables such as reproduction_rate and num_new_cases are represented as functions which compute the CPD of these variables and return a distribution object.
These CPD functions are marked with the `beanmachine.ppl.random_variable` decorator and all EPDs are based on PyTorch EPDs in the `torch.distribution` package.

```
import beanmachine.ppl as bm
import torch.distribution as dist

num_infected = 1087980

@bm.random_variable
def reproduction_rate():
    return Exponential(0.1)

@bm.random_variable
def num_new_cases():
    return Poisson(reproduction_rate() *  num_infected)
```

The call to inference involves first creating an appropriate inference engine object and then invoking the `infer` method as follows

```
engine = bm.<Inference Engine>()
engine.infer(queries = [reproduction_rate()],
             observations = {new_new_cases() : 238154})
```

Here it is important to note that the call `reproduction_rate()` in the list of queried variables refers to the name of a random variable while the same call in the CPD of `num_new_cases()` refers to the value of the random variable.
 -->
