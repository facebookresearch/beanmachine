Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Bean Machine Graph is a C++-based library for statistical inference over stochastic computation graphs.

Bean Machine PPL is a probabilistic programming language for inference over statistical models written in the Python language using a declarative syntax.

## Build

Clone https://github.com/facebookincubator/BeanMachine.git

cd beanmachine

./build/fbcode_builder/getdeps.py build beanmachine

If running from inside Facebook as a VM, use the following command.
sudo http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 ./build/fbcode_builder/getdeps.py build beanmachine
