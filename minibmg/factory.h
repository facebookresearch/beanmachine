/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/type.h"

namespace beanmachine::minibmg {

class Graph::Factory {
 public:
  uint add_constant(double value);

  uint add_operator(enum Operator op, std::vector<uint> parents);

  // returns the index of the query in the samples
  uint add_query(uint parent);

  uint add_variable(const std::string& name, const uint variable_index);

  inline const Node* operator[](uint node_id) const {
    return nodes[node_id];
  }
  Graph build();
  ~Factory();

 private:
  std::vector<const Node*> nodes;
  uint next_query = 0;
};

enum Type expected_result_type(enum Operator op);
extern const std::vector<std::vector<enum Type>> expected_parents;
uint arity(Operator op);
enum Type op_type(enum Operator op);

} // namespace beanmachine::minibmg
