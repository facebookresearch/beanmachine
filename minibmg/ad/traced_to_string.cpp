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

PrintedForm print(const Traced& traced, std::map<Traced, PrintedForm>& cache) {
  auto op = traced.op();
  switch (op) {
    case Operator::CONSTANT: {
      const TracedConstant* b =
          std::static_pointer_cast<const TracedConstant, const TracedBody>(
              traced.ptr())
              .get();
      return PrintedForm{b->value.to_string(), Precedence::Term};
    }
    case Operator::VARIABLE: {
      const TracedVariable* b =
          std::static_pointer_cast<const TracedVariable, const TracedBody>(
              traced.ptr())
              .get();
      auto name =
          (b->name.length() == 0) ? fmt::format("__{0}", b->sequence) : b->name;
      return PrintedForm{name, Precedence::Term};
    }
    case Operator::ADD:
    case Operator::SUBTRACT:
    case Operator::MULTIPLY:
    case Operator::DIVIDE: {
      const TracedOp* b =
          std::static_pointer_cast<const TracedOp, const TracedBody>(
              traced.ptr())
              .get();
      auto this_precedence = precedence_for_operator(op);
      auto this_string = string_for_operator(op);
      auto left = cache[b->args[0]];
      auto ls = (left.precedence > this_precedence)
          ? fmt::format("({0})", left.string)
          : left.string;
      auto right = cache[b->args[1]];
      auto rs = (right.precedence >= this_precedence)
          ? fmt::format("({0})", right.string)
          : right.string;
      return PrintedForm{
          fmt::format("{0} {1} {2}", ls, this_string, rs), this_precedence};
    }
    case Operator::NEGATE: {
      const TracedOp* b =
          std::static_pointer_cast<const TracedOp, const TracedBody>(
              traced.ptr())
              .get();
      auto this_precedence = precedence_for_operator(op);
      auto right = cache[b->args[0]];
      auto rs = (right.precedence >= this_precedence)
          ? fmt::format("({0})", right.string)
          : right.string;
      return PrintedForm{fmt::format("-{0}", rs), this_precedence};
    }
    case Operator::POW:
    case Operator::POLYGAMMA: {
      const TracedOp* b =
          std::static_pointer_cast<const TracedOp, const TracedBody>(
              traced.ptr())
              .get();
      auto left = cache[b->args[0]];
      auto ls = (left.precedence > Precedence::Term)
          ? fmt::format("({0})", left.string)
          : left.string;
      auto right = cache[b->args[1]];
      auto rs = right.string;
      auto this_string = string_for_operator(op);
      return PrintedForm{
          fmt::format("{0}.{1}({2})", ls, this_string, rs), Precedence::Term};
    }
    case Operator::EXP:
    case Operator::LOG:
    case Operator::ATAN:
    case Operator::LGAMMA: {
      const TracedOp* b =
          std::static_pointer_cast<const TracedOp, const TracedBody>(
              traced.ptr())
              .get();
      auto this_string = string_for_operator(op);
      auto left = cache[b->args[0]];
      auto ls = (left.precedence > Precedence::Term)
          ? fmt::format("({0})", left.string)
          : left.string;
      return PrintedForm{
          fmt::format("{0}.{1}()", ls, this_string), Precedence::Term};
    }
    case Operator::IF_LESS:
    case Operator::IF_EQUAL: {
      const TracedOp* b =
          std::static_pointer_cast<const TracedOp, const TracedBody>(
              traced.ptr())
              .get();
      auto this_string = string_for_operator(op);
      auto left = cache[b->args[0]];
      auto ls = (left.precedence > Precedence::Term)
          ? fmt::format("({0})", left.string)
          : left.string;
      return PrintedForm{
          fmt::format(
              "{0}.{1}({2}, {3}, {4})",
              ls,
              this_string,
              cache[b->args[1]].string,
              cache[b->args[2]].string,
              cache[b->args[3]].string),
          Precedence::Term};
    }
    default: {
      return PrintedForm{
          "std::string to_string(const Traced& traced) not implemented",
          Precedence::Term};
    }
  }
}

PrintedForm to_printed_form(
    const Traced& traced,
    std::map<Traced, PrintedForm> cache) {
  auto p = print(traced, cache);
  const Traced& t = traced;
  cache[t] = p;
  return p;
}

std::string to_string(const Traced& traced) {
  auto successors = [](const Traced& t) {
    switch (t.op()) {
      case Operator::CONSTANT:
      case Operator::VARIABLE:
        return std::vector<Traced>{};
      default:
        return static_cast<const TracedOp*>(t.ptr().get())->args;
    }
  };
  auto pred_counts = count_predecessors<Traced>({traced}, successors);
  std::map<Traced, uint> pred_counts_copy = pred_counts;
  std::vector<Traced> topologically_sorted;
  bool sorted = topological_sort<Traced>(
      pred_counts_copy, successors, topologically_sorted);
  if (!sorted) {
    throw std::invalid_argument("cycle in traced expression");
  }
  std::reverse(topologically_sorted.begin(), topologically_sorted.end());
  std::map<Traced, PrintedForm> cache{};
  for (auto n : topologically_sorted) {
    auto p = print(n, cache);
    cache[n] = p;
  }

  return cache[traced].string;
}

} // namespace

std::string beanmachine::minibmg::Traced::to_string() const {
  const Traced& self = *this;
  return ::to_string(self);
}
