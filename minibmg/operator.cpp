/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/operator.h"
#include <folly/Format.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

using namespace beanmachine::minibmg;

std::unordered_map<Operator, std::string> operator_names;
std::unordered_map<std::string, Operator> string_to_operator;

bool _c0 = [] {
  auto add = [](Operator op, const std::string& name) {
    if (operator_names.contains(op)) {
      throw std::logic_error(fmt::format(
          "beanmachine::minibmg::operator_names duplicate operator name for {0}",
          op));
    }
    operator_names[op] = name;
    string_to_operator[name] = op;
  };
  add(Operator::NO_OPERATOR, "NO_OPERATOR");
  add(Operator::CONSTANT, "CONSTANT");
  add(Operator::VARIABLE, "VARIABLE");
  add(Operator::ADD, "ADD");
  add(Operator::SUBTRACT, "SUBTRACT");
  add(Operator::NEGATE, "NEGATE");
  add(Operator::MULTIPLY, "MULTIPLY");
  add(Operator::DIVIDE, "DIVIDE");
  add(Operator::POW, "POW");
  add(Operator::EXP, "EXP");
  add(Operator::LOG, "LOG");
  add(Operator::ATAN, "ATAN");
  add(Operator::LGAMMA, "LGAMMA");
  add(Operator::POLYGAMMA, "POLYGAMMA");
  add(Operator::IF_EQUAL, "IF_EQUAL");
  add(Operator::IF_LESS, "IF_LESS");
  add(Operator::DISTRIBUTION_NORMAL, "DISTRIBUTION_NORMAL");
  add(Operator::DISTRIBUTION_HALF_NORMAL, "DISTRIBUTION_HALF_NORMAL");
  add(Operator::DISTRIBUTION_BETA, "DISTRIBUTION_BETA");
  add(Operator::DISTRIBUTION_BERNOULLI, "DISTRIBUTION_BERNOULLI");
  add(Operator::SAMPLE, "SAMPLE");

  // check that we have set the name for every operator.
  for (Operator op = Operator::NO_OPERATOR; op < Operator::LAST_OPERATOR;
       op = (Operator)((int)op + 1)) {
    if (!operator_names.contains(op)) {
      throw std::logic_error(fmt::format(
          "beanmachine::minibmg::operator_names missing operator name for {0}",
          op));
    }
  }

  return true;
}();

} // namespace

namespace beanmachine::minibmg {

Operator operator_from_name(const std::string& name) {
  auto found = string_to_operator.find(name);
  if (found != string_to_operator.end()) {
    return found->second;
  }

  return Operator::NO_OPERATOR;
}

std::string to_string(Operator op) {
  auto found = operator_names.find(op);
  if (found != operator_names.end()) {
    return found->second;
  }

  return "NO_OPERATOR";
}

} // namespace beanmachine::minibmg
