/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph.h"
#include <fmt/core.h>
#include <folly/json.h>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include "beanmachine/minibmg/dedup.h"
#include "beanmachine/minibmg/factory.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

const std::vector<Nodep> roots(
    const std::vector<Nodep>& queries,
    const std::list<std::pair<Nodep, double>>& observations) {
  std::list<Nodep> roots;
  for (auto& n : queries) {
    roots.push_back(n);
  }
  for (auto& p : observations) {
    if (p.first->op != Operator::SAMPLE) {
      throw std::invalid_argument(fmt::format("can only observe a sample"));
    }
    roots.push_front(p.first);
  }
  std::vector<Nodep> all_nodes;
  if (!topological_sort<Nodep>(roots, &in_nodes, all_nodes)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(all_nodes.begin(), all_nodes.end());
  return all_nodes;
}

struct QueriesAndObservations {
  std::vector<Nodep> queries;
  std::list<std::pair<Nodep, double>> observations;
  ~QueriesAndObservations() {}
};

} // namespace

namespace beanmachine::minibmg {

template <>
class DedupHelper<QueriesAndObservations> {
 public:
  std::vector<Nodep> find_roots(const QueriesAndObservations& qo) const {
    std::vector<Nodep> roots;
    for (auto& q : qo.observations) {
      roots.push_back(q.first);
    }
    for (auto& n : qo.queries) {
      roots.push_back(n);
    }
    return roots;
  }
  QueriesAndObservations rewrite(
      const QueriesAndObservations& qo,
      const std::unordered_map<Nodep, Nodep>& map) const {
    DedupHelper<std::vector<Nodep>> h1{};
    DedupHelper<std::list<std::pair<Nodep, double>>> h2{};
    return QueriesAndObservations{
        h1.rewrite(qo.queries, map), h2.rewrite(qo.observations, map)};
  }
};

using dynamic = folly::dynamic;

Graph Graph::create(
    const std::vector<Nodep>& queries,
    const std::list<std::pair<Nodep, double>>& observations) {
  for (auto& p : observations) {
    if (p.first->op != Operator::SAMPLE) {
      throw std::invalid_argument(fmt::format("can only observe a sample"));
    }
  }

  auto qo0 = QueriesAndObservations{queries, observations};
  auto qo1 = dedup(qo0);

  std::vector<Nodep> all_nodes = roots(qo1.queries, qo1.observations);
  Graph::validate(all_nodes);
  return Graph{all_nodes, qo1.queries, qo1.observations};
}

Graph::~Graph() {}

Graph::Graph(
    const std::vector<Nodep>& nodes,
    const std::vector<Nodep>& queries,
    const std::list<std::pair<Nodep, double>>& observations)
    : nodes{nodes}, queries{queries}, observations{observations} {}

void Graph::validate(std::vector<Nodep> nodes) {
  std::unordered_set<Nodep> seen;
  // Check the nodes.
  for (int i = 0, n = nodes.size(); i < n; i++) {
    assert(!nodes.empty()); // quiet, lint
    auto& node = nodes[i];

    // TODO: improve the exception diagnostics on failure.  e.g. how to identify
    // a node?

    // Check that the operator is in range.
    if (node->op < (Operator)0 || node->op >= Operator::LAST_OPERATOR) {
      throw std::invalid_argument(
          fmt::format("Node {0} has invalid operator {1}", i, (int)node->op));
    }

    // Check the node type
    if (node->type != expected_result_type(node->op)) {
      throw std::invalid_argument(fmt::format(
          "Node {0} has type {1} but should be {2}",
          i,
          to_string(node->type),
          to_string(expected_result_type(node->op))));
    }

    // Check the predecessor nodes
    switch (node->op) {
      case Operator::CONSTANT:
      case Operator::VARIABLE:
        break;

      case Operator::SAMPLE: {
        auto op = std::dynamic_pointer_cast<const SampleNode>(node);
        Nodep distribution = op->distribution;
        if (!seen.count(distribution)) {
          throw std::invalid_argument(
              fmt::format("Node {0} has a parent not previously seen", i));
        }
        if (distribution->type != Type::DISTRIBUTION) {
          throw std::invalid_argument(fmt::format(
              "Node {0} (SAMPLE) should have a DISTRIBUTION input", i));
        }
        break;
      }

      // Check other operators.
      default: {
        auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
        unsigned ix = (unsigned)node->op;
        auto parent_types = expected_parents[ix];
        if (op->in_nodes.size() != parent_types.size()) {
          throw std::invalid_argument(fmt::format(
              "Node {0} should have {1} parents", i, parent_types.size()));
        }
        for (int j = 0, m = parent_types.size(); j < m; j++) {
          const Nodep& parent = op->in_nodes[j];
          if (!seen.count(parent)) {
            throw std::invalid_argument(
                fmt::format("Node {0} has a parent not previously seen", i));
          }
          if (parent->type != parent_types[j]) {
            throw std::invalid_argument(fmt::format(
                "Node {0} should have a {1} input",
                i,
                to_string(parent_types[j])));
          }
        }
        break;
      }
    }

    seen.insert(node);
  }
}

folly::dynamic graph_to_json(const Graph& g) {
  std::unordered_map<Nodep, unsigned long> node_to_identifier;
  dynamic result = dynamic::object;
  result["comment"] = "created by graph_to_json";
  dynamic a = dynamic::array;

  unsigned long next_identifier = 0;
  for (auto& node : g) {
    // assign node identifiers sequentially.  They are called "sequence" in the
    // generated json.
    auto identifier = next_identifier++;
    node_to_identifier[node] = identifier;
    dynamic dyn_node = dynamic::object;
    dyn_node["sequence"] = identifier;
    dyn_node["operator"] = to_string(node->op);
    dyn_node["type"] = to_string(node->type);
    switch (node->op) {
      case Operator::CONSTANT: {
        auto n = std::dynamic_pointer_cast<const ConstantNode>(node);
        dyn_node["value"] = n->value;
        break;
      }
      case Operator::VARIABLE: {
        auto n = std::dynamic_pointer_cast<const VariableNode>(node);
        dyn_node["name"] = n->name;
        dyn_node["identifier"] = n->identifier;
        break;
      }
      case Operator::SAMPLE: {
        auto n = std::dynamic_pointer_cast<const SampleNode>(node);
        dynamic in_nodes = dynamic::array;
        in_nodes.push_back(node_to_identifier[n->distribution]);
        dyn_node["in_nodes"] = in_nodes;
        break;
      }
      default: {
        auto n = std::dynamic_pointer_cast<const OperatorNode>(node);
        dynamic in_nodes = dynamic::array;
        for (auto& pred : n->in_nodes) {
          in_nodes.push_back(node_to_identifier[pred]);
        }
        dyn_node["in_nodes"] = in_nodes;
        break;
      }
    }
    a.push_back(dyn_node);
  }
  result["nodes"] = a;

  dynamic observations = dynamic::array;
  for (auto& q : g.observations) {
    dynamic d = dynamic::object;
    auto id = node_to_identifier[q.first];
    d["node"] = id;
    d["value"] = q.second;
    observations.push_back(d);
  }
  result["observations"] = observations;

  dynamic queries = dynamic::array;
  for (auto& q : g.queries) {
    queries.push_back(node_to_identifier[q]);
  }
  result["queries"] = queries;

  return result;
}

JsonError::JsonError(const std::string& message) : message(message) {}

Graph json_to_graph(folly::dynamic d) {
  Graph::Factory gf;
  // Nodes are identified by a "sequence" number appearing in json.
  // They are arbitrary numbers.  The only requirement is that they
  // are distinct.  They are used to identify nodes in the json.
  // This map is used to identify the specific node when it is
  // referenced in the json.
  std::unordered_map<int, Nodep> identifier_to_node;

  auto json_nodes = d["nodes"];
  if (!json_nodes.isArray()) {
    throw JsonError("missing \"nodes\" property");
  }
  for (auto json_node : json_nodes) {
    auto identifierv = json_node["sequence"];
    if (!identifierv.isInt()) {
      throw JsonError("missing sequence number.");
    }
    auto identifier = identifierv.asInt();

    auto opv = json_node["operator"];
    if (!opv.isString()) {
      throw JsonError("missing operator.");
    }
    auto op = operator_from_name(opv.asString());
    if (op == Operator::NO_OPERATOR) {
      throw JsonError("bad operator " + opv.asString());
    }

    Type type;
    auto typev = json_node["type"];
    if (!typev.isString()) {
      type = op_type(op);
    } else {
      type = type_from_name(typev.asString());
    }

    Nodep node;
    switch (op) {
      case Operator::CONSTANT: {
        auto valuev = json_node["value"];
        double value;
        if (valuev.isInt()) {
          value = valuev.asInt();
        } else if (valuev.isDouble()) {
          value = valuev.asDouble();
        } else {
          throw JsonError("bad value for constant.");
        }
        if (type != Type::REAL) {
          throw JsonError("bad type for constant.");
        }
        node = std::make_shared<const ConstantNode>(value);
        break;
      }
      case Operator::VARIABLE: {
        auto namev = json_node["name"];
        std::string name = "";
        if (namev.isString()) {
          name = namev.asString();
        } else {
          throw JsonError("bad name for variable.");
        }
        if (type != Type::REAL) {
          throw JsonError("bad type for variable.");
        }
        auto variable_indexv = json_node["variable_index"];
        if (!variable_indexv.isInt()) {
          throw JsonError("bad variable_index for variable.");
        }
        auto variable_index = (unsigned)variable_indexv.asInt();
        node = std::make_shared<const VariableNode>(name, variable_index);
        break;
      }
      case Operator::SAMPLE: {
        auto in_nodesv = json_node["in_nodes"];
        if (!in_nodesv.isArray()) {
          throw JsonError("missing in_nodes.");
        }
        if (in_nodesv.size() != 1) {
          throw JsonError("sample requires one input node.");
        }
        auto in_nodev = in_nodesv[0];
        if (!in_nodev.isInt()) {
          throw JsonError("missing in_node for operator.");
        }
        auto in_node_i = in_nodev.asInt();
        if (!identifier_to_node.contains(in_node_i)) {
          throw JsonError("bad in_node for operator.");
        }
        auto in_node = identifier_to_node[in_node_i];
        auto rvid = fmt::format("S{}", identifier);
        node = std::make_shared<const SampleNode>(in_node, rvid);
        break;
      }
      default: {
        auto in_nodesv = json_node["in_nodes"];
        if (!in_nodesv.isArray()) {
          throw JsonError("missing in_nodes.");
        }
        std::vector<Nodep> in_nodes;
        for (auto& in_nodev : in_nodesv) {
          if (!in_nodev.isInt()) {
            throw JsonError("missing in_node for operator.");
          }
          auto in_node_i = in_nodev.asInt();
          if (!identifier_to_node.contains(in_node_i)) {
            throw JsonError("bad in_node for operator.");
          }
          auto in_node = identifier_to_node[in_node_i];
          in_nodes.push_back(in_node);
        }
        node = std::make_shared<const OperatorNode>(in_nodes, op, type);
        break;
      }
    }

    if (identifier_to_node.contains(identifier)) {
      throw JsonError(fmt::format("duplicate node ID {}.", identifier));
    }
    identifier_to_node[identifier] = node;
  }

  std::vector<Nodep> queries;
  auto query_nodes = d["queries"];
  if (query_nodes.isArray()) {
    for (auto& query : query_nodes) {
      if (!query.isInt()) {
        throw JsonError("bad query value.");
      }
      auto query_i = query.asInt();
      if (!identifier_to_node.contains(query_i)) {
        throw JsonError(fmt::format("bad in_node {} for query.", query_i));
      }
      auto query_node = identifier_to_node[query_i];
      queries.push_back(query_node);
    }
  }

  std::list<std::pair<Nodep, double>> observations;
  auto observation_nodes = d["observations"];
  if (observation_nodes.isArray()) {
    for (auto& obs : observation_nodes) {
      auto node = obs["node"];
      if (!node.isInt()) {
        throw JsonError("bad observation node.");
      }
      auto node_i = node.asInt();
      if (!identifier_to_node.contains(node_i)) {
        throw JsonError(fmt::format("bad in_node {} for observation.", node_i));
      }
      auto& obs_node = identifier_to_node[node_i];
      auto& value = obs["value"];
      if (!node.isDouble() && !node.isInt()) {
        throw JsonError("bad value for observation.");
      }
      auto value_d = value.asDouble();
      observations.push_back(std::pair{obs_node, value_d});
    }
  }

  return Graph::create(queries, observations);
}

} // namespace beanmachine::minibmg
