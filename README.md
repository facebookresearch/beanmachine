# Bean Machine

[![Lint](https://github.com/facebookresearch/beanmachine/actions/workflows/lint.yml/badge.svg)](https://github.com/facebookresearch/beanmachine/actions/workflows/lint.yml)
[![Tests](https://github.com/facebookresearch/beanmachine/actions/workflows/test.yml/badge.svg)](https://github.com/facebookresearch/beanmachine/actions/workflows/test.yml)


## Overview

Bean Machine is a probabilistic programming language for inference over statistical models written in the Python language using a declarative syntax. Bean Machine is built on top of PyTorch and Bean Machine Graph, a custom C++ backend.
Check out [our tutorials and Quick Start](https://beanmachine.org/docs/quickstart) to get started!

## Installation
Bean Machine supports Python 3.7+ and PyTorch 1.10.

### Install the Latest Release with Pip

<!-- TODO: replace this command with a regular pip install after OSS -->
```bash
pip install --extra-index-url https://test.pypi.org/simple/ beanmachine
```

### Install from Source

To download the latest Bean Machine source code from GitHub:

```bash
git clone https://github.com/facebookresearch/beanmachine.git
cd beanmachine
```

Then, you can choose from any of the following installation options.

#### Anaconda

We recommend using [conda](https://docs.conda.io/en/latest/) to manage the virtual environment and install the necessary build dependencies.

```bash
conda create -n {env name} python=3.7; conda activate {env name}
conda install boost eigen
pip install .
```

#### Docker

```bash
docker build -t beanmachine .
docker run -it beanmachine:latest bash
```

#### Validate Installation

If you would like to run the builtin unit tests:

```bash
pip install pytest
pytest .
```
