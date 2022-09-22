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
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/minibmg.h"
#include "beanmachine/minibmg/topological.h"

using namespace beanmachine::minibmg;

// Private helper data structures and functions used in producing a string form
// for a Traced.
namespace {

enum class Precedence {
  Term,
  Product,
  Sum,
};

struct PrintedForm {
  PrintedForm() : string{""}, precedence{Precedence::Term} {}
  PrintedForm(std::string string, Precedence precedence)
      : string{string}, precedence{precedence} {}
  PrintedForm(const PrintedForm& other)
      : string{other.string}, precedence{other.precedence} {}
  PrintedForm& operator=(const PrintedForm& other) = default;
  ~PrintedForm() {}
  std::string string;
  Precedence precedence;
};

Precedence precedence_for_operator(Operator op) {
  switch (op) {
    case Operator::ADD:
    case Operator::SUBTRACT:
    case Operator::NEGATE:
      return Precedence::Sum;
    case Operator::MULTIPLY:
    case Operator::DIVIDE:
      return Precedence::Product;
    default:
      return Precedence::Term;
  }
}

std::string string_for_operator(Operator op) {
  switch (op) {
    case Operator::ADD:
      return "+";
    case Operator::SUBTRACT:
    case Operator::NEGATE:
      return "-";
    case Operator::MULTIPLY:
      return "*";
    case Operator::DIVIDE:
      return "/";
    case Operator::POW:
      return "pow";
    case Operator::POLYGAMMA:
      return "polygamma";
    case Operator::EXP:
      return "exp";
    case Operator::LOG:
      return "log";
    case Operator::ATAN:
      return "atan";
    case Operator::LGAMMA:
      return "lgamma";
    case Operator::IF_EQUAL:
      return "if_equal";
    case Operator::IF_LESS:
      return "if_less";
    default:
      return "<program error>";
  }
}

PrintedForm print(Nodep node, std::map<Nodep, PrintedForm>& cache) {
  auto op = node->op;
  switch (op) {
    case Operator::CONSTANT: {
      auto n = std::dynamic_pointer_cast<const ConstantNode>(node);
      return PrintedForm{fmt::format("{}", n->value), Precedence::Term};
    }
    case Operator::VARIABLE: {
      auto n = std::dynamic_pointer_cast<const VariableNode>(node);
      auto name = (n->name.length() == 0) ? fmt::format("__{0}", n->identifier)
                                          : n->name;
      return PrintedForm{name, Precedence::Term};
    }
    case Operator::ADD:
    case Operator::SUBTRACT:
    case Operator::MULTIPLY:
    case Operator::DIVIDE: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto this_precedence = precedence_for_operator(op);
      auto this_string = string_for_operator(op);
      auto left = cache[n->in_nodes[0]];
      auto ls = (left.precedence > this_precedence)
          ? fmt::format("({0})", left.string)
          : left.string;
      auto right = cache[n->in_nodes[1]];
      auto rs = (right.precedence >= this_precedence)
          ? fmt::format("({0})", right.string)
          : right.string;
      return PrintedForm{
          fmt::format("{0} {1} {2}", ls, this_string, rs), this_precedence};
    }
    case Operator::NEGATE: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto this_precedence = precedence_for_operator(op);
      auto right = cache[n->in_nodes[0]];
      auto rs = (right.precedence >= this_precedence)
          ? fmt::format("({0})", right.string)
          : right.string;
      return PrintedForm{fmt::format("-{0}", rs), this_precedence};
    }
    case Operator::POW:
    case Operator::POLYGAMMA: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto left = cache[n->in_nodes[0]];
      auto ls = left.string;
      auto right = cache[n->in_nodes[1]];
      auto rs = right.string;
      auto this_string = string_for_operator(op);
      return PrintedForm{
          fmt::format("{1}({0}, {2})", ls, this_string, rs), Precedence::Term};
    }
    case Operator::EXP:
    case Operator::LOG:
    case Operator::ATAN:
    case Operator::LGAMMA: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto this_string = string_for_operator(op);
      auto left = cache[n->in_nodes[0]];
      auto ls = left.string;
      return PrintedForm{
          fmt::format("{1}({0})", ls, this_string), Precedence::Term};
    }
    case Operator::IF_LESS:
    case Operator::IF_EQUAL: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto this_string = string_for_operator(op);
      auto left = cache[n->in_nodes[0]];
      auto ls = left.string;
      return PrintedForm{
          fmt::format(
              "{1}({0}, {2}, {3}, {4})",
              ls,
              this_string,
              cache[n->in_nodes[1]].string,
              cache[n->in_nodes[2]].string,
              cache[n->in_nodes[3]].string),
          Precedence::Term};
    }
    default: {
      return PrintedForm{
          fmt::format(
              "{}:{}: not implemented: to_string(const Traced& traced)",
              __FILE__,
              __LINE__),
          Precedence::Term};
    }
  }
}

} // namespace

namespace beanmachine::minibmg {

std::string to_string(const Nodep& node);

std::string to_string(const Traced& traced) {
  return to_string(traced.node);
}

std::string to_string(const Nodep& node) {
  auto successors = [](const Nodep& t) -> std::vector<Nodep> {
    switch (t->op) {
      case Operator::CONSTANT:
      case Operator::VARIABLE:
        return {};
      default:
        return std::dynamic_pointer_cast<const OperatorNode>(t)->in_nodes;
    }
  };
  auto pred_counts = count_predecessors<Nodep>({node}, successors);
  std::map<Nodep, unsigned> pred_counts_copy = pred_counts;
  std::vector<Nodep> topologically_sorted;
  bool sorted = topological_sort<Nodep>(
      pred_counts_copy, successors, topologically_sorted);
  if (!sorted) {
    throw std::invalid_argument("cycle in graph");
  }
  std::reverse(topologically_sorted.begin(), topologically_sorted.end());
  std::map<Nodep, PrintedForm> cache{};
  for (auto n : topologically_sorted) {
    auto p = print(n, cache);
    cache[n] = p;
  }

  return cache[node].string;
}

} // namespace beanmachine::minibmg
