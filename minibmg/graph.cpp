/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph.h"
#include <fmt/core.h>
#include <folly/json.h>
#include <unordered_map>
#include "beanmachine/minibmg/factory.h"

namespace beanmachine::minibmg {

using dynamic = folly::dynamic;

Graph::Graph(std::vector<const Node*> nodes) : nodes{nodes} {}

Graph::~Graph() {
  for (auto node : nodes) {
    delete node;
  }
}

Graph Graph::create(std::vector<const Node*> nodes) {
  Graph::validate(nodes);
  return Graph{nodes};
}

void Graph::validate(std::vector<const Node*> nodes) {
  std::unordered_set<const Node*> seen;
  unsigned next_query = 0;
  // Check the nodes.
  for (int i = 0, n = nodes.size(); i < n; i++) {
    auto node = nodes[i];

    // TODO: check that node identifiers are unique

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
        break;
      case Operator::QUERY: {
        const QueryNode* q = (QueryNode*)node;
        if (q->query_index != next_query) {
          throw std::invalid_argument(fmt::format(
              "Node {0} has query index {1} but should be {2}",
              i,
              q->query_index,
              next_query));
        }
        next_query++;
        if (!seen.count(q->in_node)) {
          throw std::invalid_argument(
              fmt::format("Query Node {0} parent not previously seen", i));
        }
        if (q->in_node->type == Type::DISTRIBUTION) {
          throw std::invalid_argument(fmt::format(
              "Query Node {0} should have a distribution input", i));
        }
        break;
      }

      // Check other operators.
      default: {
        const OperatorNode* op = (OperatorNode*)node;
        unsigned ix = (unsigned)node->op;
        auto parent_types = expected_parents[ix];
        if (op->in_nodes.size() != parent_types.size()) {
          throw std::invalid_argument(fmt::format(
              "Node {0} should have {1} parents", i, parent_types.size()));
        }
        for (int j = 0, m = parent_types.size(); j < m; j++) {
          const Node* parent = op->in_nodes[j];
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
      }
    }

    seen.insert(node);
  }
}

folly::dynamic graph_to_json(const Graph& g) {
  std::unordered_map<const Node*, unsigned long> id_map{};
  dynamic result = dynamic::object;
  result["comment"] = "created by graph_to_json";
  dynamic a = dynamic::array;

  unsigned long next_identifier = 0;
  for (auto node : g) {
    // assign node identifiers sequentially.  They are called "sequence" in the
    // generated json.
    auto identifier = next_identifier++;
    id_map[node] = identifier;
    dynamic dyn_node = dynamic::object;
    dyn_node["sequence"] = identifier;
    dyn_node["operator"] = to_string(node->op);
    dyn_node["type"] = to_string(node->type);
    switch (node->op) {
      case Operator::QUERY: {
        auto n = (const QueryNode*)node;
        dyn_node["query_index"] = n->query_index;
        dyn_node["in_node"] = id_map[n->in_node];
        break;
      }
      case Operator::CONSTANT: {
        auto n = (const ConstantNode*)node;
        dyn_node["value"] = n->value;
        break;
      }
      default: {
        auto n = (const OperatorNode*)node;
        dynamic in_nodes = dynamic::array;
        for (auto pred : n->in_nodes) {
          in_nodes.push_back(id_map[pred]);
        }
        dyn_node["in_nodes"] = in_nodes;
        break;
      }
    }
    a.push_back(dyn_node);
  }

  result["nodes"] = a;
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
  std::unordered_map<int, const Node*> identifier_to_node;
  std::vector<const Node*> all_nodes;

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

    Node* node;
    switch (op) {
      case Operator::QUERY: {
        auto query_indexv = json_node["query_index"];
        if (!query_indexv.isInt()) {
          throw JsonError("missing query_index for query.");
        }
        auto query_index = (unsigned)query_indexv.asInt();

        auto in_nodev = json_node["in_node"];
        if (!in_nodev.isInt()) {
          throw JsonError("missing in_node for query.");
        }
        auto in_node_i = in_nodev.asInt();
        if (identifier_to_node.find(in_node_i) == identifier_to_node.end()) {
          throw JsonError("bad in_node for query.");
        }
        auto in_node = identifier_to_node.find(in_node_i)->second;
        if (type != Type::NONE) {
          throw JsonError("bad type for query.");
        }
        node = new QueryNode{query_index, in_node};
        break;
      }
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
          throw JsonError("bad type for query.");
        }
        node = new ConstantNode{value};
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
        node = new VariableNode{name, variable_index};
        break;
      }
      default: {
        auto in_nodesv = json_node["in_nodes"];
        if (!in_nodesv.isArray()) {
          throw JsonError("missing in_nodes.");
        }
        std::vector<const Node*> in_nodes;
        for (auto in_nodev : in_nodesv) {
          if (!in_nodev.isInt()) {
            throw JsonError("missing in_node for query.");
          }
          auto in_node_i = in_nodev.asInt();
          if (identifier_to_node.find(in_node_i) == identifier_to_node.end()) {
            throw JsonError("bad in_node for query.");
          }
          auto in_node = identifier_to_node.find(in_node_i)->second;
          in_nodes.push_back(in_node);
        }
        node = new OperatorNode{in_nodes, op, type};
        break;
      }
    }

    if (identifier_to_node.find(identifier) != identifier_to_node.end()) {
      throw JsonError(fmt::format("duplicate node ID {}.", identifier));
    }
    identifier_to_node[identifier] = node;
    all_nodes.push_back(node);
  }

  return Graph::create(all_nodes);
}

} // namespace beanmachine::minibmg
