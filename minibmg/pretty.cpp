/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/pretty.h"
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/topological.h"

// Private helper data structures and functions used in producing a string form
// for a Traced.
namespace {

using namespace beanmachine::minibmg;

enum class Precedence {
  Term,
  Product,
  Sum,
};

struct PrintedForm {
  std::string string;
  Precedence precedence;

  PrintedForm() : string{""}, precedence{Precedence::Term} {}
  PrintedForm(std::string string, Precedence precedence)
      : string{string}, precedence{precedence} {}
  PrintedForm(const PrintedForm& other)
      : string{other.string}, precedence{other.precedence} {}
  PrintedForm& operator=(const PrintedForm& other) = default;
  ~PrintedForm() {}
};

Precedence precedence_for_operator(Operator op) {
  switch (op) {
    case Operator::ADD:
    case Operator::SUBTRACT:
      return Precedence::Sum;
    case Operator::MULTIPLY:
    case Operator::DIVIDE:
      return Precedence::Product;
    case Operator::NEGATE:
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
      const auto& name = (n->name.length() == 0)
          ? fmt::format("__{0}", n->identifier)
          : n->name;
      return PrintedForm{name, Precedence::Term};
    }
    case Operator::ADD:
    case Operator::SUBTRACT:
    case Operator::MULTIPLY:
    case Operator::DIVIDE: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto this_precedence = precedence_for_operator(op);
      auto operator_name = string_for_operator(op);
      auto left = cache[n->in_nodes[0]];
      auto ls = (left.precedence > this_precedence)
          ? fmt::format("({0})", left.string)
          : left.string;
      auto right = cache[n->in_nodes[1]];
      const auto& rs = (right.precedence >= this_precedence)
          ? fmt::format("({0})", right.string)
          : right.string;
      return PrintedForm{
          fmt::format("{0} {1} {2}", ls, operator_name, rs), this_precedence};
    }
    case Operator::NEGATE: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto this_precedence = precedence_for_operator(op);
      auto right = cache[n->in_nodes[0]];
      auto rs = (right.precedence > this_precedence)
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
      auto operator_name = string_for_operator(op);
      return PrintedForm{
          fmt::format("{0}({1}, {2})", operator_name, ls, rs),
          Precedence::Term};
    }
    case Operator::EXP:
    case Operator::LOG:
    case Operator::ATAN:
    case Operator::LGAMMA: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto operator_name = string_for_operator(op);
      auto left = cache[n->in_nodes[0]];
      auto ls = left.string;
      return PrintedForm{
          fmt::format("{0}({1})", operator_name, ls), Precedence::Term};
    }
    case Operator::IF_LESS:
    case Operator::IF_EQUAL: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto operator_name = string_for_operator(op);
      auto left = cache[n->in_nodes[0]];
      auto ls = left.string;
      return PrintedForm{
          fmt::format(
              "{0}({1}, {2}, {3}, {4})",
              operator_name,
              ls,
              cache[n->in_nodes[1]].string,
              cache[n->in_nodes[2]].string,
              cache[n->in_nodes[3]].string),
          Precedence::Term};
    }
    case Operator::DISTRIBUTION_BERNOULLI: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      return PrintedForm{
          fmt::format("{0}({1})", "bernoulli", cache[n->in_nodes[0]].string),
          Precedence::Term};
    }
    case Operator::DISTRIBUTION_BETA: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      return PrintedForm{
          fmt::format(
              "{0}({1}, {2})",
              "beta",
              cache[n->in_nodes[0]].string,
              cache[n->in_nodes[1]].string),
          Precedence::Term};
    }
    case Operator::DISTRIBUTION_HALF_NORMAL: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      return PrintedForm{
          fmt::format("{0}({1})", "half_normal", cache[n->in_nodes[0]].string),
          Precedence::Term};
    }
    case Operator::DISTRIBUTION_NORMAL: {
      auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
      return PrintedForm{
          fmt::format(
              "{0}({1}, {2})",
              "normal",
              cache[n->in_nodes[0]].string,
              cache[n->in_nodes[1]].string),
          Precedence::Term};
    }
    case Operator::SAMPLE: {
      auto n = std::dynamic_pointer_cast<const SampleNode>(node);
      return PrintedForm{
          fmt::format(
              "{0}({1}, \"{2}\")",
              "sample",
              cache[n->distribution].string,
              n->rvid),
          Precedence::Term};
    }
    default: {
      return PrintedForm{
          fmt::format(
              "{}:{}: not implemented: to_string(const Traced& traced) for operator {}",
              __FILE__,
              __LINE__,
              to_string(op)),
          Precedence::Term};
    }
  }
}

} // namespace

namespace beanmachine::minibmg {

// Pretty-print a set of Nodes.  Returns a PrettyResult.
const PrettyResult pretty_print(std::vector<Nodep> roots) {
  std::map<Nodep, unsigned> counts =
      count_predecessors<Nodep>({roots.begin(), roots.end()}, in_nodes);
  // count the roots as uses too, so that they get put into a variable if
  // also used elsewhere.
  for (auto n : roots) {
    counts[n]++;
  }

  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::logic_error("nodes have a cycle");
  }
  reverse(sorted.begin(), sorted.end());

  std::map<Nodep, PrintedForm> cache;
  std::vector<std::string> prelude;
  unsigned next_temp = 1;
  for (auto n : sorted) {
    PrintedForm p = print(n, cache);
    // We do not dedup constants and variables in the printed form, as the
    // printed form is simpler without doing do.
    if (counts[n] > 1 && n->op != Operator::VARIABLE &&
        n->op != Operator::CONSTANT) {
      std::string temp = fmt::format("temp_{}", next_temp++);
      std::string assignment = fmt::format("auto {} = {};", temp, p.string);
      prelude.push_back(assignment);
      p = {temp, Precedence::Term};
    }
    cache[n] = p;
  }

  std::unordered_map<Nodep, std::string> code;
  for (auto p : cache) {
    code[p.first] = p.second.string;
  }

  return PrettyResult{prelude, code};
}

// Pretty-print a graph into the code that would need to be written using the
// fluid factory to reproduce it.  Assumes the graph has already been
// deduplicated.
std::string pretty_print(const Graph& graph) {
  std::vector<Nodep> roots;
  for (auto p : graph.observations) {
    roots.push_back(p.first);
  }
  for (auto n : graph.queries) {
    roots.push_back(n);
  }
  auto pretty_result = pretty_print(roots);
  std::stringstream code;
  for (auto p : pretty_result.prelude) {
    code << p << std::endl;
  }
  code << "Graph::FluentFactory fac;" << std::endl;
  for (auto q : graph.queries) {
    code << fmt::format("fac.query({});", pretty_result.code[q]) << std::endl;
  }
  for (auto o : graph.observations) {
    code << fmt::format(
                "fac.observe({}, {});", pretty_result.code[o.first], o.second)
         << std::endl;
  }

  return code.str();
}

} // namespace beanmachine::minibmg
