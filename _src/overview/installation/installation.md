---
id: installation
title: 'Installation'
sidebar_label: 'Installation'
---
<!-- @import "../../header.md" -->

## Did You Check Out Colab?
The Google Colaboratory web service (Colab) is probably the quickest way to run Bean Machine. For example, [here is what our Coin Flipping tutorial looks like on Colab](https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Coin_flipping.ipynb). Similar links can be found for each of our tutorials in the Tutorials section.

## Requirements
Python 3.7-3.8 and PyTorch 1.10.

Note: Some features are not yet supported on Python 3.9+

## Latest Release

Using `pip` you can get the latest release with the following command:
```
pip install beanmachine
```

## Installing From Source
To install from source, the first step is to clone the git repository:
```
git clone https://github.com/facebookresearch/beanmachine.git
cd beanmachine
pip install -e .
```
If you are a developer and plan to experiment with modifying the code, we recommend replacing the last step above with:
```
pip install -e ".[dev]"
```
