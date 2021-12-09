---
slug: '/tutorials'
title: 'Tutorials'
sidebar_label: 'Tutorials'
---
import useBaseUrl from '@docusaurus/useBaseUrl';

These Bean Machine tutorials demonstrate various types of statistical models that users can build in Bean Machine.

## Tutorials

### Coin Flipping

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Coin_flipping.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Coin_flipping.ipynb)

This tutorial demonstrates modeling and running inference on a simple coin-flipping model in Bean Machine. This should offer an accessible introduction to fundamental features of Bean Machine.


### Linear Regression

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Linear_Regression.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Linear_Regression.ipynb)

This tutorial demonstrates modeling and running inference on a simple univariate linear regression model in Bean Machine. This should offer an accessible introduction to models that use PyTorch tensors and Newtonian Monte Carlo inference in Bean Machine. It will also teach you effective practices for prediction on new datasets with Bean Machine.


### Logistic Regression

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Bayesian_Logistic_Regression.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Bayesian_Logistic_Regression.ipynb)

This tutorial shows how to build a simple Bayesian model to deduce the line which separates two categories of points.


### Sparse Logistic Regression

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Tutorial_Implement_sparse_logistic_regression.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Tutorial_Implement_sparse_logistic_regression.ipynb)

This tutorial demonstrates modeling and running inference on a sparse logistic regression model in Bean Machine. This tutorial showcases the inference techniques in Bean Machine, and applies the model to a public dataset to evaluate performance. This tutorial will also introduce the `@bm.functional` decorator, which can be used to deterministically transform random variables. This tutorial uses it for convenient post-processing.


### Modeling Radon Decay Using a Hierarchical Regression with Continuous Data

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_regression.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_regression.ipynb)

This tutorial explores linear regression in combination with hierarchical priors. We will be using data from Gelman and Hill on radon levels found in buildings in Minnesota; [Hill J and Gelman A](https://pdfs.semanticscholar.org/e6d6/8a23f02485cfbb3e35d6bba862a682a2f160.pdf). This tutorial shows how  to prepare data for running a hierarchical regression with Bean Machine, how to run inference on that regression, and how to use ArviZ diagnostics to understand what Bean Machine is doing.


### Modeling MLB Performance Using Hierarchical Modeling with Repeated Binary Trials

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_modeling.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_modeling.ipynb)

This tutorial demonstrates the application of hierarchical models with data from the 1970 season of [Major League Baseball (MLB)](https://en.wikipedia.org/wiki/Major_League_Baseball) found in the paper by [Efron and Morris 1975](http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/MultilevelData/EfronMorrisJASA1975.pdf). In addition to teaching effective hierarchical modeling techniques for binary data, this tutorial will explore how you can use different pooling techniques to enable strength-borrowing between observations.


### Modeling NBA Foul Calls Using Item Response Theory

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Item_Response_Theory.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Item_Response_Theory.ipynb)

This tutorial demonstrates how to use Bean Machine to predict when NBA players will receive a foul call from a referee. This model and exposition is based on [Austin Rochford's 2018 analysis](https://austinrochford.com/posts/2018-02-04-nba-irt-2.html) of the [2015/2016 NBA season games](https://www.basketball-reference.com/leagues/NBA_2016_games.html) data. It will introduce you to Item Response Theory, and demonstrate its advantages over standard regression models. It will also iterate on that model several times, demonstrating how to evolve your model to improve predictive performance.


### Modeling Medical Efficacy by Marginalizing Discrete Variables in Zero-Inflated Count Data

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Zero_inflated_count_data.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Zero_inflated_count_data.ipynb)

This tutorial investigates data that originated from [Berry](https://www.jstor.org/stable/2531826), and was analyzed by [Farewell and Sprott](https://www.jstor.org/stable/2531746), from a study about the efficacy of a medication that helps prevent irregular heartbeats. Counts of patients' irregular heartbeats were observed 60 seconds before the administration of the drug, and 60 seconds after the medication was taken. A large percentage of records show zero irregular heartbeats in the 60 seconds after taking the medication. There are more observed zeros than would be expected if we were to sample from one of the common statistical discrete distributions. The problem we face is trying to model these zero counts in order to appropriately quantify the medication's impact on reducing irregular heartbeats.


### Hidden Markov Model

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Hidden_Markov_model.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hidden_Markov_model.ipynb)

This tutorial demonstrates modeling and running inference on a hidden Markov model (HMM) in Bean Machine. The flexibility of this model allows us to demonstrate useful features of Bean Machine, including `CompositionalInference`, multi-site inference, and posterior predictive checks. This model makes use of discrete latent states, and shows how Bean Machine can easily run inference for models comprised of both discrete and continuous latent variables.


### Gaussian Mixture Model

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/GMM_with_2_dimensions_and_4_components.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/GMM_with_2_dimensions_and_4_components.ipynb)

This tutorial uses Bean Machine to infer which latent clusters observed points are drawn from. It uses a 2-dimensional Gaussian mixture model with 4 mixture components, and shows how Bean Machine can automatically recover the means and variances of these latent components.

[Mixture models](https://en.wikipedia.org/wiki/Mixture_model) are useful in problems where individuals from multiple sub-populations are aggregated together. A common use case for GMMs is unsupervised clustering, where one seeks to infer which sub-population an individual belongs without any labeled training data. Using Bean Machine, we provide a Bayesian treatment of this problem and infer a posterior distribution over cluster parameters and cluster assignments of observations.


### Neal's Funnel

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Neals_funnel.ipynb) • [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hidden_Markov_model.ipynb)

This tutorial demonstrates modeling and running inference on the Neal's funnel model in Bean Machine. Neal's funnel is a synthetic model in which the posterior distribution is known beforehand, and Bean Machine's inference engine is tasked with automatically recovering that posterior distribution. Neal's funnel has proven difficult-to-handle for classical inference methods due to its unusual topology. This tutorial demonstrates how to overcome this by using second-order gradient methods in Bean Machine. It also demonstrates how to implement models with factors in Bean Machine through custom distributions.
