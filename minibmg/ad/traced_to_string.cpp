/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/minibmg.h"
#include "beanmachine/minibmg/pretty.h"
#include "beanmachine/minibmg/topological.h"

namespace beanmachine::minibmg {

std::string to_string(const Nodep& node) {
  auto pretty_result = pretty_print({node});
  std::stringstream code;
  for (auto p : pretty_result.prelude) {
    code << p << std::endl;
  }
  code << pretty_result.code[node];
  return code.str();
}

std::string to_string(const Traced& traced) {
  return to_string(traced.node);
}

} // namespace beanmachine::minibmg
