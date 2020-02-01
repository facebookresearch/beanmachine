// Copyright (c) Facebook, Inc. and its affiliates.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace beanmachine {
namespace ppl {
namespace tensorops {

/*
Compute the first and the second gradient of the output Tensor
w.r.t. the input Tensor.

:param output: A Tensor variable with a single element.
:param input: A 1-d tensor input variable that was used to compute the
              output. Note: the input must have requires_grad=True
:returns: tuple of Tensor variables -- The first and the second gradient.
*/
std::tuple<torch::Tensor, torch::Tensor> gradients(
    torch::Tensor output,
    torch::Tensor input) {
  // based on: https://github.com/pytorch/pytorch/issues/32045
  pybind11::gil_scoped_release no_gil;
  if (output.numel() != 1) {
    throw std::invalid_argument(
        "output tensor must have exactly one element, got " +
        std::to_string(output.numel()));
  }
  // compute the first gradient
  torch::Tensor grad1 =
      torch::autograd::grad({output}, {input}, {}, true, true, true)[0];
  // for each element of the first gradient compute the second gradient
  grad1 = grad1.reshape(-1);
  int64_t input_len = grad1.size(0);
  torch::Tensor hessian = torch::empty(
        {input_len, input_len}, {}, torch::TensorOptions().dtype(input.dtype()));
  for (int64_t i = 0; i < input_len; i++) {
    hessian[i] = (torch::autograd::grad({grad1[i]}, {input}, {}, true, true, true)[0]).reshape(-1);
  }

  return std::make_tuple(grad1, hessian);
}

PYBIND11_MODULE(tensorops, module) {
  module.doc() = "module for operations on tensors";

  module.def(
      "gradients",
      &gradients,
      "Compute the first and the second gradient of the output Tensor w.r.t. the input Tensor.",
      pybind11::arg("output"),
      pybind11::arg("input"));
}

} // namespace tensorops
} // namespace ppl
} // namespace beanmachine
