/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/type.h"
#include <string>
#include <unordered_map>

namespace {

using namespace beanmachine::minibmg;

std::unordered_map<Type, std::string> type_names;
std::unordered_map<std::string, Type> string_to_type;

bool _c1 = [] {
  auto add = [](Type type, const std::string& name) {
    type_names[type] = name;
    string_to_type[name] = type;
  };
  add(Type::REAL, "REAL");
  add(Type::DISTRIBUTION, "DISTRIBUTION");
  return true;
}();

} // namespace

namespace beanmachine::minibmg {

Type type_from_name(const std::string& name) {
  auto found = string_to_type.find(name);
  if (found != string_to_type.end()) {
    return found->second;
  }

  return Type::NONE;
}

std::string to_string(Type type) {
  auto found = type_names.find(type);
  if (found != type_names.end()) {
    return found->second;
  }

  return "NONE";
}

} // namespace beanmachine::minibmg
