/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/rewriters/localopt.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/rewriters/constant_fold.h"
#include "beanmachine/minibmg/rewriters/update_children.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

// The following meta-variables are used as placeholders in the patterns for
// unification with nodes to be transformed.
auto a = Traced::variable("a", 1);
auto b = Traced::variable("b", 2);
auto c = Traced::variable("c", 3);
auto d = Traced::variable("d", 4);
auto w = Traced::variable("w", 5);
auto x = Traced::variable("x", 6);
auto y = Traced::variable("y", 7);
auto z = Traced::variable("z", 8);

// meta-variables whose name starts with a k are constrained to match scalar
// constants.
auto k = Traced::variable("k", 10);
auto k1 = Traced::variable("k1", 11);
auto k2 = Traced::variable("k2", 12);
auto k3 = Traced::variable("k3", 13);
auto k4 = Traced::variable("k4", 14);

using Transform = std::pair<Traced, Traced>;

std::vector<Transform> transforms = {
    {0 + x, x},
    {x + 0, x},
    {x + x, 2 * x},
    {a + x + x, a + 2 * x},
    {y * x + x, (y + 1) * x},
    {a + y * x + x, a + (y + 1) * x},
    {y * x + z * x, (y + z) * x},
    {a + y * x + z * x, a + (y + z) * x},
    {a + (-b), a - b},

    {0 - x, -x},
    {x - 0, x},
    {x - x, 0},
    {a - k1 - k2, a - (k1 + k2)},
    {a - x - x, a - 2 * x},
    {a - b * x - x, a - (b + 1) * x},
    {a - b * x - c * x, a - (b + c) * x},
    {a - x + b * x, a + (b - 1) * x},
    {a - k1 / x - k2 / x, a - (k1 + k2) / x},
    {-(k1 / x) - k2 / x, -(k1 + k2) / x},
    {(k1 / x) - k2 / x, (k1 - k2) / x},

    {-(-x), x},

    {0 * x, 0},
    {x * 0, 0},
    {1 * x, x},
    {x * 1, x},
    {k1 * (k2 * x), (k1 * k2) * x},
    {x * (-1), -x},
    {(-1) * x, -x},
    {(-x) * y, -(x* y)},
    {x * (-y), -(x* y)},
    {x * (1 / y), x / y},
    {(1 / y) * x, x / y},
    {(-x) * y, -(x* y)},
    {x * (-y), -(x* y)},
    {k1 * (k2 / a), (k1 * k2) / a},
    {(-k) * x, -(k* x)},

    {0 / x, 0},
    {x / k, (1 / k) * x},
    {x / (x / y), y},
    {(-x) / y, -(x / y)},
    {x / (-y), -(x / y)},
    {x / y / y, x* pow(y, -2)},

    {pow(x, 0), 1},
    {pow(0, x), 0},
    {pow(x, 1), x},

    {exp(log(x)), x},
    {log(exp(x)), x}};

std::map<std::type_index, std::vector<Transform>> transforms_by_node_type(
    const std::vector<Transform>& transforms) {
  std::map<std::type_index, std::vector<Transform>> result{};
  for (auto& transform : transforms) {
    const Node* key = transform.first.node.get();
    std::type_index type_index = typeid(*key);
    auto found = result.find(type_index);
    if (found == result.end()) {
      result[type_index] = {transform};
    } else {
      found->second.push_back(transform);
    }
  }

  return result;
}

std::map<std::type_index, std::vector<Transform>> transforms_by_operator =
    transforms_by_node_type(transforms);

NodepValueEquals same{};

using Environment = std::unordered_map<Nodep, Nodep>;

bool unify(const Nodep& pattern, const Nodep& value, Environment& environment) {
  if (auto var = std::dynamic_pointer_cast<const ScalarVariableNode>(pattern)) {
    auto found = environment.find(var);
    if (found == environment.end()) {
      if (var->name.starts_with("k") &&
          !std::dynamic_pointer_cast<const ScalarConstantNode>(value)) {
        // a variable whose name starts with k must match a constant.
        return false;
      }
      // update the environment
      environment.insert({var, value});
      return true;
    } else {
      return same(found->second, value);
    }
  } else if (std::dynamic_pointer_cast<const ScalarSampleNode>(pattern)) {
    throw std::logic_error("sample nodes should not appear in patterns");
  } else if (
      auto konst =
          std::dynamic_pointer_cast<const ScalarConstantNode>(pattern)) {
    auto kvalue = std::dynamic_pointer_cast<const ScalarConstantNode>(value);
    if (!kvalue) {
      return false;
    }
    auto k1 = konst->constant_value;
    auto k2 = kvalue->constant_value;
    return k1 == k2 || (std::isnan(k1) && std::isnan(k2));
  } else {
    // Check that the top-level operator of the input is the same as that of the
    // pattern.
    if (typeid(*pattern.get()) != typeid(*value.get())) {
      return false;
    }

    // other operators (other than constant, variable, and sample) have no data
    // other than their inputs.  Check their inputs.
    auto pattern_inputs = in_nodes(pattern);
    auto value_inputs = in_nodes(value);
    auto n = pattern_inputs.size();
    if (n != value_inputs.size()) {
      throw std::logic_error(
          "a given node type should have a fixed number of inputs");
    }
    for (int i = 0; i < n; i++) {
      if (!unify(pattern_inputs[i], value_inputs[i], environment)) {
        return false;
      }
    }

    return true;
  }
}

class ReplacementInterpolator;
Nodep apply_transforms(Nodep node);

// This class constructs the replacement once a pattern has been matched
class ReplacementInterpolator : NodeEvaluatorVisitor<Traced> {
  const Environment& environment;

 public:
  explicit ReplacementInterpolator(const Environment& environment)
      : NodeEvaluatorVisitor<Traced>{}, environment{environment} {}

  ScalarNodep interpolate(const ScalarNodep& replacement, bool topmost) {
    visited_node = replacement;
    replacement->accept(*this);
    ScalarNodep result = this->result.node;
    this->result = Traced{nullptr};
    return topmost
        ? result
        : std::dynamic_pointer_cast<const ScalarNode>(apply_transforms(result));
  }

 private:
  ScalarNodep visited_node;
  void visit(const ScalarVariableNode* node) override {
    Nodep n = visited_node;
    auto found = environment.find(n);
    if (found == environment.end()) {
      throw std::logic_error(
          fmt::format("variable {} not found in the environment", node->name));
    }
    auto replacement =
        std::dynamic_pointer_cast<const ScalarNode>(found->second);
    this->result = Traced{replacement};
  }
  void visit(const ScalarSampleNode* node) override {
    throw std::logic_error("replacements should not contain samples");
  }
  Traced evaluate_input(const ScalarNodep& node) override {
    return Traced{interpolate(node, false)};
  }
  std::shared_ptr<const Distribution<Traced>> evaluate_input_distribution(
      const DistributionNodep& node) override {
    throw std::logic_error("replacements should not contain distributions");
  }
};

Nodep apply_one_transform(Nodep node) {
  std::type_index type_index = typeid(*node.get());
  auto found = transforms_by_operator.find(type_index);
  if (found == transforms_by_operator.end()) {
    return node;
  }

  std::vector<Transform>& transforms = found->second;
  Environment environment{};
  for (auto& transform : transforms) {
    environment.clear();
    auto scalar_node = std::dynamic_pointer_cast<const ScalarNode>(node);
    auto [pattern, replacement] = transform;
    if (scalar_node && unify(pattern.node, scalar_node, environment)) {
      return ReplacementInterpolator{environment}.interpolate(
          replacement.node, true);
    }
  }

  return node;
}

Nodep apply_transforms(Nodep node) {
  Nodep new_node = node;
  while (true) {
    Nodep previous = new_node;
    new_node = constant_fold(new_node);
    new_node = apply_one_transform(new_node);
    if (previous == new_node) {
      break;
    }
  }

  return new_node;
}

} // namespace

namespace beanmachine::minibmg {

std::unordered_map<Nodep, Nodep> opt_map(std::vector<Nodep> roots) {
  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());

  // a value-based, map, which treats semantically identical nodes as the same.
  NodeNodeValueMap map;

  // We also build a map that uses object (pointer) identity to find elements,
  // so that clients are not using recursive node equality tests.
  std::unordered_map<Nodep, Nodep> identity_map;

  for (auto& node : sorted) {
    auto new_node = node;
    new_node = update_children(new_node, identity_map);

    if (auto found = map.find(node); found != map.end()) {
      new_node = found->second;
    } else {
      new_node = apply_transforms(new_node);
      map.insert(node, new_node);
    }

    identity_map.insert({node, new_node});
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
