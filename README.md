# Bean Machine
<div align="center">
  <a href="http://beanmachine.org"> <img width="220px" height="220px" src="https://beanmachine.org/img/beanmachine.svg"></a>
</div>

[![Lint](https://github.com/facebookresearch/beanmachine/actions/workflows/lint.yml/badge.svg)](https://github.com/facebookresearch/beanmachine/actions/workflows/lint.yml)
[![Tests](https://github.com/facebookresearch/beanmachine/actions/workflows/test.yml/badge.svg)](https://github.com/facebookresearch/beanmachine/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/beanmachine)](https://pypi.org/project/beanmachine)


## Overview

Bean Machine is a probabilistic programming language for inference over statistical models written in the Python language using a declarative syntax. Bean Machine is built on top of PyTorch and Bean Machine Graph, a custom C++ backend.
Check out our [tutorials](https://beanmachine.org/docs/tutorials/) and [Quick Start](https://beanmachine.org/docs/overview/quick_start/) to get started!

## Installation
Bean Machine supports Python 3.7-3.9 and PyTorch 1.10.

### Install the Latest Release with Pip

```bash
python -m pip install beanmachine
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
conda create -n {env name} python=3.8; conda activate {env name}
conda install -c conda-forge boost-cpp eigen=3.4.0
python -m pip install .
```

#### Docker

```bash
docker build -t beanmachine .
docker run -it beanmachine:latest bash
```

#### Validate Installation

If you would like to run the builtin unit tests:

```bash
python -m pip install "beanmachine[test]"
pytest src
```

## License
Bean Machine is MIT licensed, as found in the [LICENSE](LICENSE) file.
