/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <random>
#include "beanmachine/minibmg/topological.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

// namespace {

struct Node {
  inline Node() {}
  std::set<Node*> successors;
};

const int n = 100;

bool is_topologically_sorted(std::vector<Node*> nodes) {
  std::set<Node*> seen;
  for (int i = nodes.size() - 1; i >= 0; i--) {
    auto n = nodes[i];
    for (auto succ : n->successors) {
      if (!seen.contains(succ))
        return false;
    }
    seen.insert(n);
  }

  return true;
}

// } // namespace

TEST(topological_test, ensure_sorted) {
  std::mt19937 gen;
  std::uniform_int_distribution<int> u(0, n - 1);

  for (int k = 0; k < 5; k++) { // run this test 5 times
    // create some nodes
    std::vector<Node*> nodes;
    for (int i = 0; i < n; i++) {
      nodes.push_back(new Node);
    }
    ASSERT_EQ(nodes.size(), n);

    // add some edges - always from left to right so there can be no cycle.
    int nedges = (int)std::pow(n, 1.5);
    for (int i = 0; i < nedges; i++) {
      while (true) {
        int l = u(gen), r = u(gen);
        if (l == r)
          continue;
        if (l > r)
          std::swap(l, r);
        auto lnode = nodes[l];
        auto rnode = nodes[r];
        std::set<Node*>& succ = lnode->successors;
        if (succ.contains(rnode))
          continue; // edge already exists; try again
        succ.insert(rnode);
        break;
      }
    }

    // shuffle the nodes so that the sort cannot tell which order they should be
    // in.
    ASSERT_EQ(nodes.size(), n);
    std::shuffle(nodes.begin(), nodes.end(), gen);
    ASSERT_EQ(nodes.size(), n);

    // topologically sort them.
    std::vector<Node*> result;
    auto sorted = topological_sort<Node*>(
        std::list<Node*>{nodes.begin(), nodes.end()},
        [](Node* node) {
          return std::list<Node*>{
              node->successors.begin(), node->successors.end()};
        },
        result);
    ASSERT_TRUE(sorted);
    ASSERT_EQ(result.size(), n);
    std::vector<Node*> vec{result.begin(), result.end()};
    ASSERT_EQ(vec.size(), n);
    ASSERT_TRUE(is_topologically_sorted(vec));

    // now add an edge that, with high probability, induces a cycle.
    vec[vec.size() - 1]->successors.insert(vec[0]);

    // shuffle the nodes so that the sort cannot tell which order they should be
    // in.
    nodes.clear();
    std::copy(vec.begin(), vec.end(), std::back_inserter(nodes));
    ASSERT_EQ(nodes.size(), n);
    std::shuffle(nodes.begin(), nodes.end(), gen);

    // topologically sort them.  if there was a cycle, this should return false.
    result.clear();
    sorted = topological_sort<Node*>(
        std::list<Node*>{nodes.begin(), nodes.end()},
        [](Node* node) {
          return std::list<Node*>{
              node->successors.begin(), node->successors.end()};
        },
        result);
    ASSERT_FALSE(sorted);

    // dispose the nodes
    for (Node* n : nodes) {
      delete n;
    }
  }
}
