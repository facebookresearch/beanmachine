/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <iostream>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/out_nodes_reflexive_transitive_closure.h"

using namespace ::testing;

using namespace beanmachine;
using namespace std;
using namespace graph;
using namespace util;

void run_ortc_test(
    const char* name,
    const Graph& graph,
    NodeID node_id,
    std::function<pair<bool, bool>(Node*)> prune,
    std::function<pair<bool, bool>(Node*)> abort,
    bool success,
    vector<NodeID> expected_ids) {
  cout << "Testing " << name << endl;
  auto ortc =
      OutNodesReflexiveTransitiveClosure(graph.get_node(node_id), prune, abort);
  ASSERT_EQ(ortc.success(), success);
  auto actual_ids = graph.get_node_ids(ortc.get_result());
  ASSERT_EQ(actual_ids, expected_ids);
}

TEST(out_nodes_reflexive_transitive_closure_test, basic_ortc) {
  /*
                         0.1    0.2
                          | \   /
                          |   +
                          |  /
                          | /
                          *
  */
  Graph g;
  auto point_one = g.add_constant_real(0.1);
  auto point_two = g.add_constant_real(0.2);
  auto sum = g.add_operator(OperatorType::ADD, {point_one, point_two});
  auto product = g.add_operator(OperatorType::MULTIPLY, {sum, point_one});

  auto prune = std::function([&](Node*) { return make_pair(false, false); });
  auto abort = std::function([&](Node*) { return make_pair(false, false); });

  auto name = "point_one"; // for error messages
  NodeID node = point_one;
  auto success = true;
  vector<NodeID> expected{point_one, sum, product};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "point_two";
  node = point_two;
  success = true;
  expected = {point_two, sum, product};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "sum";
  node = sum;
  success = true;
  expected = {sum, product};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "product";
  node = product;
  success = true;
  expected = {product};
  run_ortc_test(name, g, node, prune, abort, success, expected);
}

TEST(out_nodes_reflexive_transitive_closure_test, ortc_with_abort) {
  /*
                         n0
                      /  |  \
                    n1   n2  n3
                     \  / \
                      n4   n5
                        \  /
                         n6
  */
  Graph g;
  auto n0 = g.add_constant_real(0.1);
  auto n1 = g.add_operator(OperatorType::EXP, {n0});
  auto n2 = g.add_operator(OperatorType::EXP, {n0});
  auto n3 = g.add_operator(OperatorType::EXP, {n0});
  auto n4 = g.add_operator(OperatorType::ADD, {n1, n2});
  auto n5 = g.add_operator(OperatorType::EXP, {n2});
  auto n6 = g.add_operator(OperatorType::ADD, {n4, n5});

  auto prune = std::function([&](Node*) { return make_pair(false, false); });

  // Abort if hitting n5
  auto abort = std::function([&](Node* node) {
    auto abort_even_for_self = true;
    return make_pair(node->index == n5, abort_even_for_self);
  });

  auto name = "n0"; // for error messages
  NodeID node = n0;
  auto success = false;
  vector<NodeID> expected{n0, n1, n2, n3, n4};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n1";
  node = n1;
  success = true;
  expected = {n1, n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n2";
  node = n2;
  success = false;
  expected = {n2, n4};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n3";
  node = n3;
  success = true;
  expected = {n3};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n4";
  node = n4;
  success = true;
  expected = {n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n5";
  node = n5;
  success = false;
  expected = {};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n6";
  node = n6;
  success = true;
  expected = {n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);
}

TEST(out_nodes_reflexive_transitive_closure_test, ortc_with_prune_and_abort) {
  /*
                         n0
                      /  |  \
                    n1   n2  n3
                     \  / \
                      n4   n5
                        \  /
                         n6
  */
  Graph g;
  auto n0 = g.add_constant_real(0.1);
  auto n1 = g.add_operator(OperatorType::EXP, {n0});
  auto n2 = g.add_operator(OperatorType::EXP, {n0});
  auto n3 = g.add_operator(OperatorType::EXP, {n0});
  auto n4 = g.add_operator(OperatorType::ADD, {n1, n2});
  auto n5 = g.add_operator(OperatorType::EXP, {n2});
  auto n6 = g.add_operator(OperatorType::ADD, {n4, n5});

  // Prune n2, including itself
  auto prune = std::function([&](Node* node) {
    auto prune_even_for_self = true;
    return make_pair(node->index == n2, prune_even_for_self);
  });

  // Abort if hitting n5
  auto abort = std::function([&](Node* node) {
    auto abort_even_for_self = true;
    return make_pair(node->index == n5, abort_even_for_self);
  });

  auto name = "n0"; // for error messages
  NodeID node = n0;
  auto success = true; // succeeds because pruning n2 leads to not reaching n5
  vector<NodeID> expected{n0, n1, n3, n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n1";
  node = n1;
  success = true;
  expected = {n1, n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n2";
  node = n2;
  success = true; // succeeds because pruning n2 leads to not reaching n5
  expected = {};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n3";
  node = n3;
  success = true;
  expected = {n3};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n4";
  node = n4;
  success = true;
  expected = {n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n5";
  node = n5;
  success = false;
  expected = {};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n6";
  node = n6;
  success = true;
  expected = {n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);
}

TEST(
    out_nodes_reflexive_transitive_closure_test,
    ortc_with_prune_and_abort_but_keeping_offending_nodes) {
  /*
                         n0
                      /  |  \
                    n1   n2  n3
                     \  / \
                      n4   n5
                        \  /
                         n6
  */
  Graph g;
  auto n0 = g.add_constant_real(0.1);
  auto n1 = g.add_operator(OperatorType::EXP, {n0});
  auto n2 = g.add_operator(OperatorType::EXP, {n0});
  auto n3 = g.add_operator(OperatorType::EXP, {n0});
  auto n4 = g.add_operator(OperatorType::ADD, {n1, n2});
  auto n5 = g.add_operator(OperatorType::EXP, {n2});
  auto n6 = g.add_operator(OperatorType::ADD, {n4, n5});

  // Prune at n2 but don't discard it
  auto prune = std::function([&](Node* node) {
    auto prune_even_for_self = false;
    return make_pair(node->index == n2, prune_even_for_self);
  });

  // Abort if hitting n5 but don't discard it
  auto abort = std::function([&](Node* node) {
    auto abort_even_for_self = false;
    return make_pair(node->index == n5, abort_even_for_self);
  });

  auto name = "n0"; // for error messages
  NodeID node = n0;
  auto success = true; // succeeds because pruning n2 leads to not reaching n5
  vector<NodeID> expected{n0, n1, n2, n3, n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n1";
  node = n1;
  success = true;
  expected = {n1, n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n2";
  node = n2;
  success = true; // succeeds because pruning n2 leads to not reaching n5
  expected = {n2};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n3";
  node = n3;
  success = true;
  expected = {n3};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n4";
  node = n4;
  success = true;
  expected = {n4, n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n5";
  node = n5;
  success = false;
  expected = {n5};
  run_ortc_test(name, g, node, prune, abort, success, expected);

  name = "n6";
  node = n6;
  success = true;
  expected = {n6};
  run_ortc_test(name, g, node, prune, abort, success, expected);
}
