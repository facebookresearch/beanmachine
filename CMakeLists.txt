# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# NOTE
# ------------------------------------------------------------------------------
# The primary purpose of this CMake build configuration is C++ testing. In
# particular, it provides an entry point to the GoogleTest suite for Bean
# Machine Graph (BMG), a C++ backend for Bean Machine (BM); see
# `.github/workflows/test.yml` for its usage.
#
# This build configuration is not relevant to end users, as the BMG library is
# already compiled and linked as a Pybind11Extension when installing the BM
# Python package.

# Config
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(beanmachine-graph LANGUAGES CXX VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Dependencies
# ------------------------------------------------------------------------------
find_package(Boost REQUIRED COMPONENTS iostreams timer CONFIG)
find_package(Eigen3 REQUIRED CONFIG)
find_package(range-v3 REQUIRED CONFIG)
find_package(GTest REQUIRED CONFIG)

# Targets
# ------------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tests)
