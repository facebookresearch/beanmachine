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
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the roots to a
// corresponding node in the transitive closure of the deduplicated graph.
std::unordered_map<Nodep, Nodep> dedup_map(std::vector<Nodep> roots);

// In order to deduplicate data in a given data structure, the programmer must
// specialize this template class to locate the roots contained in
// that data structure, and to write a replacement for data structure in which
// nodes have been deduplicated.  We provide a number of specializations for
// data structures likely to be needed.
template <class T>
class DedupHelper {
 public:
  DedupHelper() = delete;
  // locate all of the roots.
  std::vector<Nodep> find_roots(const T&) const;
  // rewrite the T, given a mapping from each node to its replacement.
  T rewrite(const T&, std::unordered_map<Nodep, Nodep>) const;
};

// Rewrite a data structure by deduplicating its nodes.  The programmer must
// either provide an implementation of DedupHelper<T> or specialize it.
template <class T>
T dedup(const T& data, const DedupHelper<T>& helper = DedupHelper<T>{}) {
  auto roots = helper.find_roots(data);
  auto map = dedup_map(roots);
  return helper.rewrite(data, map);
}

// A single node can be deduplicated
template <>
class DedupHelper<Nodep> {
 public:
  std::vector<Nodep> find_roots(Nodep& n) const {
    return {n};
  }
  Nodep rewrite(const Nodep& node, const std::unordered_map<Nodep, Nodep>& map)
      const {
    auto f = map.find(node);
    return f == map.end() ? node : f->second;
  }
};

// A vector can be deduplicated.
template <class T>
class DedupHelper<std::vector<T>> {
  DedupHelper<T> t_helper{};

 public:
  std::vector<Nodep> find_roots(const std::vector<T>& roots) const {
    std::vector<Nodep> result;
    for (const auto& root : roots) {
      auto more_roots = t_helper.find_roots(root);
      result.push_back(more_roots.begin(), more_roots.end());
    }
    return result;
  }
  std::vector<T> rewrite(
      const std::vector<T>& roots,
      const std::unordered_map<Nodep, Nodep>& map) const {
    std::vector<T> result;
    for (const auto& root : roots) {
      result.push_back(t_helper.rewrite(root, map));
    }
    return result;
  }
};

// A list can be deduplicated
template <class T>
class DedupHelper<std::list<T>> {
  DedupHelper<T> t_helper{};

 public:
  std::vector<Nodep> find_roots(const std::list<T>& roots) const {
    std::vector<Nodep> result;
    for (const auto& root : roots) {
      auto more_roots = t_helper.find_roots(root);
      result.push_back(more_roots.begin(), more_roots.end());
    }
    return result;
  }
  std::list<T> rewrite(
      const std::list<T>& roots,
      const std::unordered_map<Nodep, Nodep>& map) const {
    std::list<T> result;
    for (const auto& root : roots) {
      result.push_back(t_helper.rewrite(root, map));
    }
    return result;
  }
};

// A pair can be deduplicated
template <class T, class U>
class DedupHelper<std::pair<T, U>> {
  DedupHelper<T> t_helper{};
  DedupHelper<U> u_helper{};

 public:
  std::vector<Nodep> find_roots(const std::pair<T, U>& root) const {
    std::vector<Nodep> result = t_helper(root.first);
    for (auto& r : u_helper.find_roots(root.second)) {
      result.push_back(r);
    }
    return result;
  }
  std::pair<T, U> rewrite(
      const std::pair<T, U>& root,
      const std::unordered_map<Nodep, Nodep>& map) const {
    return {
        t_helper.rewrite(root.first, map), u_helper.rewrite(root.second, map)};
  }
};

// A double can be deduplicated (no action)
template <>
class DedupHelper<double> {
 public:
  std::vector<Nodep> find_roots(const double& root) const {
    return {};
  }
  double rewrite(const double& root, std::unordered_map<Nodep, Nodep>) const {
    return root;
  }
};

// A Real (wrapper around a double) can be deduplicated (no action)
template <>
class DedupHelper<Real> {
 public:
  std::vector<Nodep> find_roots(const Real& t) const {
    return {};
  }
  // rewrite the T, given a mapping from each node to its replacement.
  Real rewrite(const Real& t, const std::unordered_map<Nodep, Nodep>& map)
      const {
    return t;
  }
};

} // namespace beanmachine::minibmg
