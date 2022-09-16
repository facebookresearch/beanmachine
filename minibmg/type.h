/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace beanmachine::minibmg {

enum class Type {
  // No type.  For example, the result of an observation or query node.
  NONE,

  // A scalar real value.
  REAL,

  // A distribution of real values.
  DISTRIBUTION,

  // Not a real type.  Used as a limit when looping through types.
  LAST_TYPE,
};

Type type_from_name(const std::string& name);
std::string to_string(Type type);

} // namespace beanmachine::minibmg
