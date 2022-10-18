/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <folly/json.h>
#include <memory>
#include <string>
#include <string_view>
#include "beanmachine/minibmg/graph2.h"

namespace {

using namespace beanmachine::minibmg;
using dynamic = folly::dynamic;

inline constexpr auto hash_djb2a(const std::string_view sv) {
  unsigned long hash{5381};
  for (unsigned char c : sv) {
    hash = ((hash << 5) + hash) ^ c;
  }
  return hash;
}

inline constexpr auto operator"" _sh(const char* str, size_t len) {
  return hash_djb2a(std::string_view{str, len});
}

class JsonNodeWriterVisitor : public Node2Visitor {
 public:
  explicit JsonNodeWriterVisitor(dynamic& dyn_node) : dyn_node{dyn_node} {}
  dynamic& dyn_node;
  void visit(const ScalarConstantNode2* node) override {
    dyn_node["operator"] = "CONSTANT";
    dyn_node["value"] = node->constant_value;
  }
  void visit(const ScalarVariableNode2* node) override {
    dyn_node["operator"] = "VARIABLE";
    dyn_node["name"] = node->name;
    dyn_node["identifier"] = node->identifier;
  }
  void visit(const ScalarSampleNode2*) override {
    dyn_node["operator"] = "SAMPLE";
  }
  void visit(const ScalarAddNode2*) override {
    dyn_node["operator"] = "ADD";
  }
  void visit(const ScalarSubtractNode2*) override {
    dyn_node["operator"] = "SUBTRACT";
  }
  void visit(const ScalarNegateNode2*) override {
    dyn_node["operator"] = "NEGATE";
  }
  void visit(const ScalarMultiplyNode2*) override {
    dyn_node["operator"] = "MULTIPLY";
  }
  void visit(const ScalarDivideNode2*) override {
    dyn_node["operator"] = "DIVIDE";
  }
  void visit(const ScalarPowNode2*) override {
    dyn_node["operator"] = "POW";
  }
  void visit(const ScalarExpNode2*) override {
    dyn_node["operator"] = "EXP";
  }
  void visit(const ScalarLogNode2*) override {
    dyn_node["operator"] = "LOG";
  }
  void visit(const ScalarAtanNode2*) override {
    dyn_node["operator"] = "ATAN";
  }
  void visit(const ScalarLgammaNode2*) override {
    dyn_node["operator"] = "LGAMMA";
  }
  void visit(const ScalarPolygammaNode2*) override {
    dyn_node["operator"] = "POLYGAMMA";
  }
  void visit(const ScalarIfEqualNode2*) override {
    dyn_node["operator"] = "IF_EQUAL";
  }
  void visit(const ScalarIfLessNode2*) override {
    dyn_node["operator"] = "IF_LESS";
  }
  void visit(const DistributionNormalNode2*) override {
    dyn_node["operator"] = "DISTRIBUTION_NORMAL";
  }
  void visit(const DistributionHalfNormalNode2*) override {
    dyn_node["operator"] = "DISTRIBUTION_HALF_NORMAL";
  }
  void visit(const DistributionBetaNode2*) override {
    dyn_node["operator"] = "DISTRIBUTION_BETA";
  }
  void visit(const DistributionBernoulliNode2*) override {
    dyn_node["operator"] = "DISTRIBUTION_BERNOULLI";
  }
};

} // namespace

namespace beanmachine::minibmg {

JsonError2::JsonError2(const std::string& message) : message(message) {}

folly::dynamic graph_to_json(const Graph2& g) {
  std::unordered_map<Node2p, unsigned long> node_to_identifier;
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
    JsonNodeWriterVisitor v{dyn_node};
    node->accept(v);
    auto in = in_nodes(node);
    if (in.size() > 0) {
      dynamic in_nodes = dynamic::array;
      for (auto& n : in) {
        in_nodes.push_back(node_to_identifier[n]);
      }
      dyn_node["in_nodes"] = in_nodes;
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

Graph2 json_to_graph2(folly::dynamic d) {
  // Nodes are identified by a "sequence" number appearing in json.
  // They are arbitrary numbers.  The only requirement is that they
  // are distinct.  They are used to identify nodes in the json.
  // This map is used to identify the specific node when it is
  // referenced in the json.
  std::unordered_map<int, Node2p> identifier_to_node;

  auto json_nodes = d["nodes"];
  if (!json_nodes.isArray()) {
    throw JsonError2("missing \"nodes\" property");
  }
  for (auto json_node : json_nodes) {
    auto identifierv = json_node["sequence"];
    if (!identifierv.isInt()) {
      throw JsonError2("missing sequence number.");
    }
    auto identifier = identifierv.asInt();

    auto opv = json_node["operator"];
    if (!opv.isString()) {
      throw JsonError2("missing operator.");
    }

    std::vector<Node2p> in_nodes{};
    switch (hash_djb2a(opv.asString())) {
      case "CONSTANT"_sh:
      case "VARIABLE"_sh:
        // in_nodes ignored
        break;
      default:
        auto in_nodesv = json_node["in_nodes"];
        if (!in_nodesv.isArray()) {
          throw JsonError2("missing in_nodes.");
        }
        for (auto& in_nodev : in_nodesv) {
          if (!in_nodev.isInt()) {
            throw JsonError2("missing in_node for operator.");
          }
          auto in_node_i = in_nodev.asInt();
          if (!identifier_to_node.contains(in_node_i)) {
            throw JsonError2("bad in_node for operator.");
          }
          auto in_node = identifier_to_node[in_node_i];
          in_nodes.push_back(in_node);
        }
        break;
    }

    Node2p node;
    switch (hash_djb2a(opv.asString())) {
      case "CONSTANT"_sh: {
        auto valuev = json_node["value"];
        double value;
        if (valuev.isInt()) {
          value = valuev.asInt();
        } else if (valuev.isDouble()) {
          value = valuev.asDouble();
        } else {
          throw JsonError2("bad value for constant.");
        }
        node = std::make_shared<const ScalarConstantNode2>(value);
      } break;
      case "VARIABLE"_sh: {
        auto namev = json_node["name"];
        std::string name = "";
        if (namev.isString()) {
          name = namev.asString();
        } else {
          throw JsonError2("bad name for variable.");
        }
        auto variable_indexv = json_node["variable_index"];
        if (!variable_indexv.isInt()) {
          throw JsonError2("bad variable_index for variable.");
        }
        auto variable_index = (unsigned)variable_indexv.asInt();
        node =
            std::make_shared<const ScalarVariableNode2>(name, variable_index);
      } break;
      case "ADD"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarAddNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "SUBTRACT"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarSubtractNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "NEGATE"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarNegateNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
      } break;
      case "MULTIPLY"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarMultiplyNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "DIVIDE"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarDivideNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "POW"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarPowNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "EXP"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarExpNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
      } break;
      case "LOG"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarLogNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
      } break;
      case "ATAN"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarAtanNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
      } break;
      case "LGAMMA"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarLgammaNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
      } break;
      case "POLYGAMMA"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarPolygammaNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "IF_EQUAL"_sh: {
        if (in_nodes.size() != 4) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarIfEqualNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[2]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[3]));
      } break;
      case "IF_LESS"_sh: {
        if (in_nodes.size() != 4) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarIfLessNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[2]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[3]));
      } break;
      case "DISTRIBUTION_NORMAL"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<DistributionNormalNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "DISTRIBUTION_HALF_NORMAL"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<DistributionHalfNormalNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
        break;

      } break;
      case "DISTRIBUTION_BETA"_sh: {
        if (in_nodes.size() != 2) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<DistributionBetaNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]),
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[1]));
      } break;
      case "DISTRIBUTION_BERNOULLI"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<DistributionBernoulliNode2>(
            std::dynamic_pointer_cast<const ScalarNode2>(in_nodes[0]));
      } break;
      case "SAMPLE"_sh: {
        if (in_nodes.size() != 1) {
          throw JsonError2("bad in_node for operator.");
        }
        node = std::make_shared<ScalarSampleNode2>(
            std::dynamic_pointer_cast<const DistributionNode2>(in_nodes[0]));
      } break;
      default:
        throw JsonError2("operator unknown: " + opv.asString());
        break;
    }

    if (identifier_to_node.contains(identifier)) {
      throw JsonError2(fmt::format("duplicate node ID {}.", identifier));
    }

    identifier_to_node[identifier] = node;
  }

  std::vector<Node2p> queries;
  auto query_nodes = d["queries"];
  if (query_nodes.isArray()) {
    for (auto& query : query_nodes) {
      if (!query.isInt()) {
        throw JsonError2("bad query value.");
      }
      auto query_i = query.asInt();
      if (!identifier_to_node.contains(query_i)) {
        throw JsonError2(fmt::format("bad in_node {} for query.", query_i));
      }
      auto query_node = identifier_to_node[query_i];
      queries.push_back(query_node);
    }
  }

  std::list<std::pair<Node2p, double>> observations;
  auto observation_nodes = d["observations"];
  if (observation_nodes.isArray()) {
    for (auto& obs : observation_nodes) {
      auto node = obs["node"];
      if (!node.isInt()) {
        throw JsonError2("bad observation node.");
      }
      auto node_i = node.asInt();
      if (!identifier_to_node.contains(node_i)) {
        throw JsonError2(
            fmt::format("bad in_node {} for observation.", node_i));
      }
      auto& obs_node = identifier_to_node[node_i];
      auto& value = obs["value"];
      if (!node.isDouble() && !node.isInt()) {
        throw JsonError2("bad value for observation.");
      }
      auto value_d = value.asDouble();
      observations.push_back(std::pair{obs_node, value_d});
    }
  }

  return Graph2::create(queries, observations);
}

} // namespace beanmachine::minibmg
