---
id: tutorials
title: 'Tutorials'
sidebar_label: 'Tutorials'
---
import useBaseUrl from '@docusaurus/useBaseUrl';

<!-- @import "../../header.md" -->

## Where to Start
Tutorials in Bean Machine demonstrate various types of statistical models that users can build in Bean Machine. If you are just getting started with Bean Machine, we recommend you work through 1 or 2 of the most basic tutorials, then consider if you might be better served by a [Package](../packages/packages.md) first, before customizing a more advanced tutorial.

### Beanstalk compiler
Additionally, consider if you can structure your statistical model in such a way that it can perform at much faster speeds, by compilation into C++. See our [Beanstalk](../beanstalk/beanstalk.md) compiler details for more information on what types of models are supported by our Bean Machine compiler.

### Beanstalk uses the Bean Machine Graph (BMG) library
With code generated that is powered by the Bean Machine Graph (BMG) library, which runs critical pieces of code in C++ rather than Python, to speed up the inference process significantly. 





-----------




Facebook specific:

 These models are also frequently used at Facebook including Team Power and Metric Ranking products (https://fb.workplace.com/notes/418250526036381) as well as new pilot studies on https://fb.quip.com/GxwQAIscFRz8 and https://fb.quip.com/UMmcAr2zczbc. Additionally, the Probabilistic Programming Languages (https://www.internalfb.com/intern/bunny/?q=group%20pplxfn) (PPL) team has collected a list of https://fb.quip.com/rrMAAuk02Jqa who can benefit from our HME methodology.

BMG: https://fb.quip.com/TDA7AIjRmScW

Ignore--saved for formatting tips:
Let's quickly translate the model we discussed in the [Introduction](../introduction/introduction.md) into Bean Machine code! Although this will get you up-and-running, **it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine**. Happy modeling!