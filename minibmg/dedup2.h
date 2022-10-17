/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the roots to a
// corresponding node in the transitive closure of the deduplicated graph.
// This is used in the implementation of dedup(), but might occasionally be
// useful in this form.
std::unordered_map<Node2p, Node2p> dedup_map(std::vector<Node2p> roots);

// In order to deduplicate data in a given data structure, the programmer must
// specialize this template class to locate the roots contained in
// that data structure, and to write a replacement for data structure in which
// nodes have been deduplicated.  We provide a number of specializations for
// data structures likely to be needed.
template <class T>
class DedupHelper2 {
 public:
  DedupHelper2() = delete;
  // locate all of the roots.
  std::vector<Node2p> find_roots(const T&) const;
  // rewrite the T, given a mapping from each node to its replacement.
  T rewrite(const T&, const std::unordered_map<Node2p, Node2p>&) const;
};

// Rewrite a data structure by deduplicating its nodes.  The programmer must
// either provide an implementation of DedupHelper2<T> or specialize it.
template <class T>
T dedup2(const T& data, const DedupHelper2<T>& helper = DedupHelper2<T>{}) {
  auto roots = helper.find_roots(data);
  auto map = dedup_map(roots);
  return helper.rewrite(data, map);
}

// A single node can be deduplicated
template <>
class DedupHelper2<Node2p> {
 public:
  std::vector<Node2p> find_roots(const Node2p& n) const {
    return {n};
  }
  Node2p rewrite(
      const Node2p& node,
      const std::unordered_map<Node2p, Node2p>& map) const {
    auto f = map.find(node);
    return f == map.end() ? node : f->second;
  }
};

// A vector can be deduplicated.
template <class T>
class DedupHelper2<std::vector<T>> {
  DedupHelper2<T> t_helper{};

 public:
  std::vector<Node2p> find_roots(const std::vector<T>& roots) const {
    std::vector<Node2p> result;
    for (const auto& root : roots) {
      auto more_roots = t_helper.find_roots(root);
      result.push_back(more_roots.begin(), more_roots.end());
    }
    return result;
  }
  std::vector<T> rewrite(
      const std::vector<T>& roots,
      const std::unordered_map<Node2p, Node2p>& map) const {
    std::vector<T> result;
    for (const auto& root : roots) {
      result.push_back(t_helper.rewrite(root, map));
    }
    return result;
  }
};

// A list can be deduplicated
template <class T>
class DedupHelper2<std::list<T>> {
  DedupHelper2<T> t_helper{};

 public:
  std::vector<Node2p> find_roots(const std::list<T>& roots) const {
    std::vector<Node2p> result;
    for (const auto& root : roots) {
      auto more_roots = t_helper.find_roots(root);
      result.push_back(more_roots.begin(), more_roots.end());
    }
    return result;
  }
  std::list<T> rewrite(
      const std::list<T>& roots,
      const std::unordered_map<Node2p, Node2p>& map) const {
    std::list<T> result;
    for (const auto& root : roots) {
      result.push_back(t_helper.rewrite(root, map));
    }
    return result;
  }
};

// A pair can be deduplicated
template <class T, class U>
class DedupHelper2<std::pair<T, U>> {
  DedupHelper2<T> t_helper{};
  DedupHelper2<U> u_helper{};

 public:
  std::vector<Node2p> find_roots(const std::pair<T, U>& root) const {
    std::vector<Node2p> result = t_helper(root.first);
    for (auto& r : u_helper.find_roots(root.second)) {
      result.push_back(r);
    }
    return result;
  }
  std::pair<T, U> rewrite(
      const std::pair<T, U>& root,
      const std::unordered_map<Node2p, Node2p>& map) const {
    return {
        t_helper.rewrite(root.first, map), u_helper.rewrite(root.second, map)};
  }
};

// A double can be deduplicated (no action)
template <>
class DedupHelper2<double> {
 public:
  std::vector<Node2p> find_roots(const double&) const {
    return {};
  }
  double rewrite(const double& root, std::unordered_map<Node2p, Node2p>) const {
    return root;
  }
};

// A Real (wrapper around a double) can be deduplicated (no action)
template <>
class DedupHelper2<Real> {
 public:
  std::vector<Node2p> find_roots(const Real&) const {
    return {};
  }
  // rewrite the T, given a mapping from each node to its replacement.
  Real rewrite(const Real& t, const std::unordered_map<Node2p, Node2p>&) const {
    return t;
  }
};

} // namespace beanmachine::minibmg
