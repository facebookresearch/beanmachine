Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

[![CircleCI](https://circleci.com/gh/facebookincubator/beanmachine.svg?style=svg&circle-token=39d1796c9ba26c78bba42dea57a9559742723be5)](https://circleci.com/gh/facebookincubator/workflows/beanmachine)

# Overview

Bean Machine is a probabilistic programming language for inference over statistical models written in the Python language using a declarative syntax.

# Installing Bean Machine
On Linux:

```bash
sudo apt-get install libboost-dev libeigen3-dev
git clone https://github.com/facebookincubator/beanmachine.git
cd beanmachine
pip install .
```

On Mac, we recommend [conda](https://docs.conda.io/en/latest/) as a package manager and [Homebrew](https://brew.sh/) to install the Xcode dependencies:

```bash
brew install llvm libomp  # Dependencies used for compiling cpp extensions

conda create -n {env name}; conda activate {env name}
conda install boost eigen
git clone https://github.com/facebookincubator/beanmachine.git
cd beanmachine
CC=clang CXX=clang++ pip install .
```

Using `docker`:

```bash
docker build -t beanmachine .
docker run -it beanmachine:latest bash
```

Further, if you would like to run the builtin unit tests:

```bash
pip install pytest
pytest .
```
