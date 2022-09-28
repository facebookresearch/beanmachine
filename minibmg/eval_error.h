/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>

namespace beanmachine::minibmg {

// Exception to throw when evaluation fails.
class EvalError : public std::exception {
 public:
  explicit EvalError(const std::string& message) : message{message} {}
  const std::string message;
};

} // namespace beanmachine::minibmg
