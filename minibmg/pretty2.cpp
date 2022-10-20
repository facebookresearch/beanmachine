/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/pretty2.h"
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node2.h"
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

class PrintedFormVisitor : public Node2Visitor {
 public:
  std::map<Node2p, PrintedForm>& cache;
  explicit PrintedFormVisitor(std::map<Node2p, PrintedForm>& cache)
      : cache{cache} {}
  PrintedForm result;
  void visit(const ScalarConstantNode2* node) override {
    result = {fmt::format("{}", node->constant_value), Precedence::Term};
  }
  void visit(const ScalarVariableNode2* node) override {
    const auto& name = (node->name.length() == 0)
        ? fmt::format("__{}", node->identifier)
        : node->name;
    result = {name, Precedence::Term};
  }
  void visit(const ScalarSampleNode2* node) override {
    auto name = fmt::format(
        "sample({}, \"{}\")", cache[node->distribution].string, node->rvid);
    result = {name, Precedence::Term};
  }
  void visit(const ScalarAddNode2* node) override {
    auto left = cache[node->left];
    auto ls = (left.precedence > Precedence::Sum)
        ? fmt::format("({})", left.string)
        : left.string;
    auto right = cache[node->right];
    const auto& rs = (right.precedence >= Precedence::Sum)
        ? fmt::format("({})", right.string)
        : right.string;
    result = {fmt::format("{} + {}", ls, rs), Precedence::Sum};
  }
  void visit(const ScalarSubtractNode2* node) override {
    auto left = cache[node->left];
    auto ls = (left.precedence > Precedence::Sum)
        ? fmt::format("({})", left.string)
        : left.string;
    auto right = cache[node->right];
    const auto& rs = (right.precedence >= Precedence::Sum)
        ? fmt::format("({})", right.string)
        : right.string;
    result = {fmt::format("{} - {}", ls, rs), Precedence::Sum};
  }
  void visit(const ScalarNegateNode2* node) override {
    auto right = cache[node->x];
    auto rs = (right.precedence > Precedence::Term)
        ? fmt::format("({})", right.string)
        : right.string;
    result = {fmt::format("-{}", rs), Precedence::Term};
  }
  void visit(const ScalarMultiplyNode2* node) override {
    auto left = cache[node->left];
    auto ls = (left.precedence > Precedence::Product)
        ? fmt::format("({})", left.string)
        : left.string;
    auto right = cache[node->right];
    const auto& rs = (right.precedence >= Precedence::Product)
        ? fmt::format("({})", right.string)
        : right.string;
    result = {fmt::format("{} * {}", ls, rs), Precedence::Product};
  }
  void visit(const ScalarDivideNode2* node) override {
    auto left = cache[node->left];
    auto ls = (left.precedence > Precedence::Product)
        ? fmt::format("({})", left.string)
        : left.string;
    auto right = cache[node->right];
    const auto& rs = (right.precedence >= Precedence::Product)
        ? fmt::format("({})", right.string)
        : right.string;
    result = {fmt::format("{} / {}", ls, rs), Precedence::Product};
  }
  void visit(const ScalarPowNode2* node) override {
    result = {
        fmt::format(
            "pow({}, {})", cache[node->left].string, cache[node->right].string),
        Precedence::Term};
  }
  void visit(const ScalarExpNode2* node) override {
    result = {fmt::format("exp({})", cache[node->x].string), Precedence::Term};
  }
  void visit(const ScalarLogNode2* node) override {
    result = {fmt::format("log({})", cache[node->x].string), Precedence::Term};
  }
  void visit(const ScalarAtanNode2* node) override {
    result = {fmt::format("atan({})", cache[node->x].string), Precedence::Term};
  }
  void visit(const ScalarLgammaNode2* node) override {
    result = {
        fmt::format("lgamma({})", cache[node->x].string), Precedence::Term};
  }
  void visit(const ScalarPolygammaNode2* node) override {
    result = {
        fmt::format(
            "polygamma({}, {})", cache[node->n].string, cache[node->x].string),
        Precedence::Term};
  }
  void visit(const ScalarIfEqualNode2* node) override {
    result = {
        fmt::format(
            "if_equal({}, {}, {}, {})",
            cache[node->a].string,
            cache[node->b].string,
            cache[node->c].string,
            cache[node->d].string),
        Precedence::Term};
  }
  void visit(const ScalarIfLessNode2* node) override {
    result = {
        fmt::format(
            "if_less({}, {}, {}, {})",
            cache[node->a].string,
            cache[node->b].string,
            cache[node->c].string,
            cache[node->d].string),
        Precedence::Term};
  }
  void visit(const DistributionNormalNode2* node) override {
    result = {
        fmt::format(
            "normal({}, {})",
            cache[node->mean].string,
            cache[node->stddev].string),
        Precedence::Term};
  }
  void visit(const DistributionHalfNormalNode2* node) override {
    result = {
        fmt::format("half_normal({})", cache[node->stddev].string),
        Precedence::Term};
  }
  void visit(const DistributionBetaNode2* node) override {
    result = {
        fmt::format(
            "beta({}, {})", cache[node->a].string, cache[node->b].string),
        Precedence::Term};
  }
  void visit(const DistributionBernoulliNode2* node) override {
    result = {
        fmt::format("bernoulli({})", cache[node->prob].string),
        Precedence::Term};
  }
};

PrintedForm print(Node2p node, PrintedFormVisitor& pfv) {
  node.get()->accept(pfv);
  return pfv.result;
}

} // namespace

namespace beanmachine::minibmg {

// Pretty-print a set of Nodes.  Returns a PrettyResult.
const Pretty2Result pretty_print(std::vector<Node2p> roots) {
  std::map<Node2p, unsigned> counts =
      count_predecessors<Node2p>({roots.begin(), roots.end()}, in_nodes);
  // count the roots as uses too, so that they get put into a variable if
  // also used elsewhere.
  for (auto n : roots) {
    counts[n]++;
  }

  std::vector<Node2p> sorted;
  if (!topological_sort<Node2p>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::logic_error("nodes have a cycle");
  }
  reverse(sorted.begin(), sorted.end());

  std::map<Node2p, PrintedForm> cache;
  PrintedFormVisitor pfv{cache};
  std::vector<std::string> prelude;
  unsigned next_temp = 1;
  for (auto n : sorted) {
    PrintedForm p = print(n, pfv);
    // We do not dedup constants and variables in the printed form, as the
    // printed form is simpler without doing do.
    if (counts[n] > 1 && !dynamic_cast<const ScalarConstantNode2*>(n.get()) &&
        !dynamic_cast<const ScalarVariableNode2*>(n.get())) {
      std::string temp = fmt::format("temp_{}", next_temp++);
      std::string assignment = fmt::format("auto {} = {};", temp, p.string);
      prelude.push_back(assignment);
      p = {temp, Precedence::Term};
    }
    cache[n] = p;
  }

  std::unordered_map<Node2p, std::string> code;
  for (auto p : cache) {
    code[p.first] = p.second.string;
  }

  return Pretty2Result{prelude, code};
}

// Pretty-print a graph into the code that would need to be written using the
// fluid factory to reproduce it.  Assumes the graph has already been
// deduplicated.
std::string pretty_print(const Graph2& graph) {
  std::vector<Node2p> roots;
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
  code << "Graph::FluidFactory fac;" << std::endl;
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

std::string pretty_print(Node2p node) {
  std::vector<Node2p> roots{node};
  auto pretty_result = pretty_print(roots);
  std::stringstream code;
  for (auto p : pretty_result.prelude) {
    code << p << std::endl;
  }
  code << pretty_result.code[node];
  return code.str();
}

} // namespace beanmachine::minibmg
