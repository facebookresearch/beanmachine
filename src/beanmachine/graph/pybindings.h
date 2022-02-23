/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// #include "beanmachine/graph/global/global_mh.h"
// #include "beanmachine/graph/global/hmc.h"
// #include "beanmachine/graph/global/nuts.h"
#include <pybind11/pybind11.h>
// #include <torch/csrc/THP.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include "beanmachine/graph/graph.h"

// to keep the linter happy this template specialization has been declared here
// in a header file that is only meant to be included by pybindings.cpp
namespace pybind11 {
namespace detail {
using namespace beanmachine::graph;

// We want NodeValues output from C++ to show up as native Python types
// such as floats, ints, bool etc. Hence we have the specific `cast` function
// in pybindings.cpp. However from Python to C++ we want to use the simple
// mappings that we will define below with `py::class_<NodeValue>` hence we
// create a super class and delegate the Python to C++ load to the base class as
// explained in this link:
// https://github.com/pybind/pybind11/issues/1176#issuecomment-343312352
template <>
struct type_caster<NodeValue> : public type_caster_base<NodeValue> {
  using base = type_caster_base<NodeValue>;

 public:
  bool load(handle src, bool convert) {
    // use base class for Python -> C++
    return base::load(src, convert);
  }

  static handle cast(NodeValue src, return_value_policy policy, handle parent) {
    return THPVariable_Wrap(src._value);
  }
};
} // namespace detail
} // namespace pybind11
