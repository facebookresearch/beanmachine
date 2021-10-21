---
slug: packages
title: 'Hierarchical Mixed Effects'
sidebar_label: 'Hierarchical Mixed Effects (HME)'
---
import useBaseUrl from '@docusaurus/useBaseUrl';

<!-- @import "../../header.md" -->

Packages in Bean Machine let a user reuse tested, proven code for specific purposes, relieving a user from needing to write their own custom Bean Machine logic.

Currently we have just one package, HME, but we encourage pull requests to add additional packages and we plan on adding additional packages as well, e.g., Gaussian Processes, in the future.

## Hierarchical Mixed Effects (HME)

Hierarchical mixed effects (HME) models are frequently used in Bayesian Statistics.

We created the HME Python package to make our current products’ code bases easier to maintain, make future statistical/ML work more efficient, and most importantly to ensure our HME methodology can be easily reused. The HME package will make hierarchical mixed effects methods widely accessible to the broader open-source community using Bean Machine.

### Fitting HME models with fixed+random effects and flexible priors
This release is the first version of our HME Python package. The package is capable of fitting Bayesian hierarchical mixed effects models with:
- any arbitrary fixed and random effects, and
- it will allow users to flexibly specify priors as they wish.

### Bean Machine Graph for faster performance
To fit hierarchical models, HME uses MCMC (Markov chain Monte Carlo) inference techniques powered by Bean Machine Graph (BMG), which runs critical pieces of code in C++ rather than Python, to speed up the inference process significantly.





-----------




Facebook specific:

 These models are also frequently used at Facebook including Team Power and Metric Ranking products (https://fb.workplace.com/notes/418250526036381) as well as new pilot studies on https://fb.quip.com/GxwQAIscFRz8 and https://fb.quip.com/UMmcAr2zczbc. Additionally, the Probabilistic Programming Languages (https://www.internalfb.com/intern/bunny/?q=group%20pplxfn) (PPL) team has collected a list of https://fb.quip.com/rrMAAuk02Jqa who can benefit from our HME methodology.

BMG: https://fb.quip.com/TDA7AIjRmScW

Ignore--saved for formatting tips:
Let's quickly translate the model we discussed in the [Introduction](../introduction/introduction.md) into Bean Machine code! Although this will get you up-and-running, **it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine**. Happy modeling!
