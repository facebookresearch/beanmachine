Bean Machine
============

[![CircleCI](https://circleci.com/gh/facebookincubator/beanmachine.svg?style=svg&circle-token=39d1796c9ba26c78bba42dea57a9559742723be5)](https://circleci.com/gh/facebookincubator/workflows/beanmachine)

# Overview

Bean Machine is a probabilistic programming language for inference over statistical models written in the Python language using a declarative syntax. Bean Machine is built on top of PyTorch and Bean Machine Graph, a custom C++ backend.
Check out [our tutorials and Quick Start](http://beanmachine.org/docs/quickstart) to get started! 

# Installing Bean Machine
Bean Machine supports Python 3.7+ and PyTorch 1.10.

**Linux**
```bash
sudo apt-get install libboost-dev libeigen3-dev
git clone https://github.com/facebookincubator/beanmachine.git
cd beanmachine
pip install .
```

**Mac OSX**

We recommend [conda](https://docs.conda.io/en/latest/) as a package manager and [Homebrew](https://brew.sh/) to install the Xcode dependencies:

```bash
brew install llvm libomp  # Dependencies used for compiling cpp extensions

conda create -n {env name}; conda activate {env name}
conda install boost eigen
git clone https://github.com/facebookincubator/beanmachine.git
cd beanmachine
CC=clang CXX=clang++ pip install .
# in dev mode
CC=clang CXX=clang++ pip install -e '.[dev]'
```

**Docker**

```bash
docker build -t beanmachine .
docker run -it beanmachine:latest bash
```

Further, if you would like to run the builtin unit tests:

```bash
pip install pytest
pytest .
```
