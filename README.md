Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

[![CircleCI](https://circleci.com/gh/facebookincubator/beanmachine.svg?style=svg&circle-token=39d1796c9ba26c78bba42dea57a9559742723be5)](https://circleci.com/gh/facebookincubator/workflows/beanmachine)

# Overview

Bean Machine is a probabilistic programming language for inference over statistical models written in the Python language using a declarative syntax.

# Installing Bean Machine
On Linux distros using the `apt` package manager

    apt-get install libboost-dev libeigen3-dev
    pip install numpy torch
    git clone https://github.com/facebookincubator/BeanMachine.git
    cd BeanMachine
    pip install .

Further, if you would like to run the builtin unit tests:

    pip install pytest
    pytest --pyargs beanmachine
