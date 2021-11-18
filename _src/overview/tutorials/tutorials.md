---
slug: '/tutorials'
title: 'Bean Machine Tutorials'
sidebar_label: 'Notebooks'
---
import useBaseUrl from '@docusaurus/useBaseUrl';

These Bean Machine tutorials demonstrate various types of statistical models that users can build in Bean Machine. If you are just getting started with Bean Machine, we recommend you work through one or two of the most basic tutorials, then consider if you might be better served by a [Package](/docs/overview/packages/packages) first, before customizing a more advanced tutorial.

### Beanstalk compiler

Additionally, consider if you can structure your statistical model in such a way that it can perform at much faster speeds, by compilation into C++. See our [Beanstalk](/docs/overview/beanstalk/beanstalk) compiler details for more information on what types of models are supported by our Bean Machine compiler.


## Tutorials

These tutorials are available as Jupyter notebooks. You can read through or download the tutorials using the *Open in GitHub* option. If you would like to run the tutorial in Google Colab, we’ve offered a link to that as well.


### Coin Flipping

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Coin_filpping.ipynb) **•** [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Coin_filpping.ipynb)

This tutorial demonstrates modeling and running inference on a simple coin-flipping model in Bean Machine. This should offer an accessible introduction to fundamental features of Bean Machine.


### Linear Regression

Open in GitHub **•** Run in Google Colab

This tutorial demonstrates modeling and running inference on a simple univariate linear regression model in Bean Machine. This should offer an accessible introduction to models that use PyTorch tensors and Newtonian Monte Carlo inference in Bean Machine. It will also teach you effective practices for prediction on new datasets with Bean Machine.


### Logistic Regression

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Bayesian_Logistic_Regression.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Bayesian_Logistic_Regression.ipynb)

The purpose of this tutorial is to show how to build a simple Bayesian model to deduce the line which separates two categories of points.


### Sparse Logistic Regression

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Tutorial_Implement_sparse_logistic_regression.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Tutorial_Implement_sparse_logistic_regression.ipynb)

This tutorial demonstrates modeling and running inference on a sparse logistic regression model in Bean Machine. This tutorial showcases the inference techniques in Bean Machine, and applies the model to a public dataset to evaluate performance. This tutorial will also introduce the `@bm.functional` decorator, which can be used to deterministically transform random variables. This tutorial uses it for convenient post-processing.


### Gaussian Mixture Model

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/GMM_with_2_dimensions_and_4_components.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/GMM_with_2_dimensions_and_4_components.ipynb)

A simple example of an open universe model is a Gaussian Mixture Model (GMM) where the number of mixture components is unknown. [Mixture models](https://en.wikipedia.org/wiki/Mixture_model) are useful in problems where individuals from multiple sub-populations are aggregated together. A common use case for GMMs is unsupervised clustering, where one seeks to infer which sub-population an individual belongs without any labeled training data. Using the bean machine open universe probabilistic programming language, we are able to provide a Bayesian treatment of this problem and draw posterior samples representing a distribution over not only the cluster assignments but also the number of clusters itself.


### Hidden Markov Model

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Hidden_Markov_model.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hidden_Markov_model.ipynb)

This tutorial demonstrates modeling and running inference on a hidden Markov model (HMM) in Bean Machine. The flexibility of this model allows us to demonstrate some of the great unique features of Bean Machine, such as block inference, compositional inference, and separation of data from the model.


### Neal's Funnel

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Tutorial_Sampling_Neal_funnel_in_Bean_Machine.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hidden_Markov_model.ipynb)

This tutorial demonstrates modeling and running inference on the so-called Neal's funnel model in Bean Machine.
Neal's funnel has proven difficult-to-handle for classical inference methods. This tutorial demonstrates how to overcome this by using second-order gradient methods in Bean Machine. It also demonstrates how to implement models with factors in Bean Machine through custom distributions.


### Robust Regression

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Robust_Linear_Regression.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Robust_Linear_Regression.ipynb)

This tutorial demonstrates modeling and running inference on a robust linear regression model in Bean Machine. This should offer a simple modification from the standard regression model to incorporate heavy tailed error models that are more robust to outliers and demonstrates modifying base models.


### **Hierarchical Modeling with Repeated Binary Trial Data**

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_modeling.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_modeling.ipynb)

In this tutorial we will demonstrate the application of hierarchical models with data from the 1970 season of [Major League Baseball (MLB)](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=d5d87c500dc6bf5a37abab41b37757804ac642b8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f66616365626f6f6b72657365617263682f6265616e6d616368696e652f643564383763353030646336626635613337616261623431623337373537383034616336343262382f7475746f7269616c732f48696572617263686963616c5f6d6f64656c696e672e6970796e623f746f6b656e3d41414c4f48354b4e4442363351374456424f554d58364442525641434b&nwo=facebookresearch%2Fbeanmachine&path=tutorials%2FHierarchical_modeling.ipynb&repository_id=201103120&repository_type=Repository#references) found in the paper by [Efron and Morris 1975](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=d5d87c500dc6bf5a37abab41b37757804ac642b8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f66616365626f6f6b72657365617263682f6265616e6d616368696e652f643564383763353030646336626635613337616261623431623337373537383034616336343262382f7475746f7269616c732f48696572617263686963616c5f6d6f64656c696e672e6970796e623f746f6b656e3d41414c4f48354b4e4442363351374456424f554d58364442525641434b&nwo=facebookresearch%2Fbeanmachine&path=tutorials%2FHierarchical_modeling.ipynb&repository_id=201103120&repository_type=Repository#references).


### **Hierarchical Regression**

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_regression.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/Hierarchical_regression.ipynb)

In this tutorial we will explore linear regression in combination with hierarchical priors. We will be using data from Gelman and Hill on radon levels found in buildings in Minnesota; [Hill J and Gelman A](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=d5d87c500dc6bf5a37abab41b37757804ac642b8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f66616365626f6f6b72657365617263682f6265616e6d616368696e652f643564383763353030646336626635613337616261623431623337373537383034616336343262382f7475746f7269616c732f48696572617263686963616c5f72656772657373696f6e2e6970796e623f746f6b656e3d41414c4f48354e4936504134354d505242464d53354c4c425256414136&nwo=facebookresearch%2Fbeanmachine&path=tutorials%2FHierarchical_regression.ipynb&repository_id=201103120&repository_type=Repository#references).


### Modeling NBA foul calls using **** **Item Response Theory**

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/NBA_IRT.ipynb) *•* Run in Google Colab

This tutorial demonstrates how to use Bean Machine to predict when NBA players will receive a foul call from a referee. This model and exposition is based on [Austin Rochford's 2018 analysis](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=d5d87c500dc6bf5a37abab41b37757804ac642b8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f66616365626f6f6b72657365617263682f6265616e6d616368696e652f643564383763353030646336626635613337616261623431623337373537383034616336343262382f7475746f7269616c732f4e42415f4952542e6970796e623f746f6b656e3d41414c4f48354c475a43454e4d554d4234324545584733425255375445&nwo=facebookresearch%2Fbeanmachine&path=tutorials%2FNBA_IRT.ipynb&repository_id=201103120&repository_type=Repository#references) of the [2015/2016 NBA season games](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=d5d87c500dc6bf5a37abab41b37757804ac642b8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f66616365626f6f6b72657365617263682f6265616e6d616368696e652f643564383763353030646336626635613337616261623431623337373537383034616336343262382f7475746f7269616c732f4e42415f4952542e6970796e623f746f6b656e3d41414c4f48354c475a43454e4d554d4234324545584733425255375445&nwo=facebookresearch%2Fbeanmachine&path=tutorials%2FNBA_IRT.ipynb&repository_id=201103120&repository_type=Repository#references) data.

### n-schools

Open in GitHub *•* Run in Google Colab

yarn-with-proxy start-fb


## Advanced

### Dynamic Bistable Hidden Markov Model

[Open in GitHub](https://github.com/facebookresearch/beanmachine/blob/master/tutorials/advanced/Dynamic_bistable_hidden_Markov_model.ipynb) *•* [Run in Google Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/master/tutorials/advanced/Dynamic_bistable_hidden_Markov_model.ipynb)

In this notebook, we will walk through performing inference for a Bistable Hidden Markov Model (HMM) with Bean Machine MCMC and compare it with handwritten Approximate Bayesian Computation (ABC) Inference.
