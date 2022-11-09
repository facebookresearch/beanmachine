/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <memory>

#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/distribution/normal.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"
#include "beanmachine/graph/third-party/nameof.h"

namespace beanmachine::graph::fluid {

/* A fluid interface for building BMG graphs. */

/*
We follow an unusual memory management strategy.

When fluid functions are called, a Node object is allocated and its
pointer is kept in a Value object.

These allocated nodes must be safely deallocated at some point.
We cannot use shared_ptr for this purpose because Graph requires unique_ptrs
and shared_ptrs cannot be converted to unique_ptrs.
We cannot use unique_ptrs because we need more than one reference to
subexpressions used in more than one Value.
We cannot use weak_ptr because those must work alongside share_ptr.
We therefore must use regular Node* but ownership and deallocation
becomes trickier.

The Nodes moved into the Graph are fine because they are converted
to unique_ptrs that will be deallocated when the Graph is destroyed.
However the user may create Nodes that are not ancestors of observations
or queries and therefore not moved into the Graph, causing a memory leak.
To prevent this, we maintain a (unfortunately, global) NodePool
that keeps all Node pointers created by fluid functions.
When a node is moved into the Graph, it is removed from the pool.
When the global pool is destroyed at the end of the process,
it deallocates all nodes not moved into the graph.

This has the potential disadvantage of keeping too many nodes in memory
if multiple graphs are built by the fluid interface.
In this case, remaining_nodes.purge() can be invoked in between
such constructions to free up memory.

This solution is less than ideal under the hood, but the user
doesn't need to worry about it except in the special cases
when 'purge' is needed.

To get rid of this solution, we would need
to refactor Graph to stop using unique_ptrs.
*/

struct NodePool {
  std::set<Node*> pool;
  void insert(Node* p) {
    pool.insert(p);
  }
  void erase(Node* p) {
    pool.erase(p);
  }
  void purge() {
    for (auto p : pool) {
      delete p;
    }
    pool.clear();
  }
  ~NodePool() {
    purge();
  }
};

extern NodePool remaining_ptrs;

Node* register_ptr(Node* p);

Node* make_node(double d);

Node* ensure_pos_real(Node* node);

Node* ensure_probability(Node* node);

struct Value {
  Node* node;
  /* implicit */ Value(Node* node) : node(node) {}
  /* implicit */ Value(double d) : node(make_node(d)) {}
};

Value beta(const Value& a, const Value& b);

Value bernoulli(const Value& p);

Value normal(const Value& mean, const Value& stddev);

Value sample(const Value& distribution);

Value operator+(const Value& a, const Value& b);

Value operator*(const Value& a, const Value& b);

inline Value operator-(const Value& a, const Value& b) {
  return a + -1 * b;
}

inline Value operator-(const Value& a) {
  return -1 * a;
}

void observe(const Value& value, const NodeValue& node_value, Graph& graph);

inline void observe(const Value& value, int node_value, Graph& graph) {
  observe(value, NodeValue(static_cast<natural_t>(node_value)), graph);
}

template <typename T>
void observe(const Value& value, const T& node_value, Graph& graph) {
  observe(value, NodeValue(node_value), graph);
}

void query(const Value& value, Graph& graph);

} // namespace beanmachine::graph::fluid
