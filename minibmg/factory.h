/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/type.h"

namespace beanmachine::minibmg {

class Graph::Factory {
 public:
  NodeId add_constant(double value);

  NodeId add_operator(enum Operator op, std::vector<NodeId> parents);

  // returns the index of the query in the samples, not a NodeId
  unsigned add_query(NodeId parent);
  unsigned add_query(NodeId parent, NodeId& new_node_id);

  NodeId add_variable(const std::string& name, const unsigned variable_index);

  inline const Node* operator[](const NodeId& node_id) const {
    auto t = nodes.find(node_id);
    if (t == nodes.end())
      return nullptr;
    return t->second;
  }
  Graph build();
  ~Factory();

 private:
  std::unordered_map<NodeId, const Node*> nodes;
  std::vector<const Node*> all_nodes;
  unsigned next_query = 0;

  NodeId add_node(const Node* node);
};

enum Type expected_result_type(enum Operator op);
extern const std::vector<std::vector<enum Type>> expected_parents;
unsigned arity(Operator op);
enum Type op_type(enum Operator op);

} // namespace beanmachine::minibmg
