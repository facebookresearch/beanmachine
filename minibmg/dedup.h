/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <concepts>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

template <class T>
class DedupAdapter;

// A concept asserting that the type `T` is a valid argument to `dedup` using
// the adapter `DDAdapter`.  In this case `T` is a type containing values of
// type `Nodep` which will undergo common subexpression elimination, and a new
// value of type `T` will be constructed and returned by `dedup`.
template <class T, class DDAdapter = DedupAdapter<T>>
concept Dedupable = requires(
    const T& t,
    const DDAdapter& a,
    const std::unordered_map<Nodep, Nodep>& map) {
  { a.find_roots(t) } -> std::convertible_to<std::vector<Nodep>>;
  { a.rewrite(t, map) } -> std::same_as<T>;
  DDAdapter();
};

// In order to deduplicate data in a given data structure, the programmer must
// specialize this template class to (1) locate the roots contained in
// that data structure, and (2) write a replacement data structure in which
// nodes (values of type Nodep) have been deduplicated.  Those methods should be
// provided by the programmer in the specialization and have the signatures of
// the two abstract methods below.  We provide a number of specializations for
// data structures likely to be needed.  `T` here is the type of the data
// structure for which nodes contained in it are to be deduplicated.
//
// The idea of organizing the code this way, with a type that the caller may
// specialize, is something I learned from the C++ standard template library.
// See std::hash<Key> and std::equal_to<Key> and their use in
// std::unordered_map.
template <class T>
class DedupAdapter {
 public:
  DedupAdapter() = delete;
  // To implement the `Dedupable` concept, you can specialize this template
  // class and provide methods with the signatures shown below.

  // locate all of the roots.
  std::vector<Nodep> find_roots(const T&) const;

  // rewrite the T, given a mapping from each node to its replacement.
  T rewrite(const T&, const std::unordered_map<Nodep, Nodep>&) const;
};

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the roots to a
// corresponding node in the transitive closure of the deduplicated graph.
// A deduplicated graph is one in which there is only one (shared) copy of
// equivalent expressions (that is, every pair of distinct nodes in the
// resulting data structure are semantically different). This is used in the
// implementation of dedup(), but might occasionally be useful to clients in
// this form.
std::unordered_map<Nodep, Nodep> dedup_map(std::vector<Nodep> roots);

// Rewrite a data structure by "deduplicating" nodes reachable from it, and
// returning a new data structure.  This is also known as common subexpression
// elimination.  The nodes in the resulting data structure will have only one
// (shared) copy of equivalent expressions (that is, every pair of distinct
// nodes in the resulting data structure are semantically different). The
// programmer must either specialize DedupAdapter<T> or provide a type to be
// used in its place.  If the input has a tree of nodes, the result may contain
// a DAG (directed acyclic graph) of nodes that is no longer a tree due to
// shared (semantically equivalent) subexpressions.
template <class T, class DDAdapter = DedupAdapter<T>>
requires Dedupable<T, DDAdapter> T
dedup(const T& data, std::unordered_map<Nodep, Nodep>* ddmap = nullptr) {
  DDAdapter adapter = DDAdapter{};
  auto roots = adapter.find_roots(data);
  auto map = dedup_map(roots);
  if (ddmap != nullptr) {
    ddmap->insert(map.begin(), map.end());
  }
  return adapter.rewrite(data, map);
}

/*
  Below here we implement many deduplication adapters for commonly used types
  ===========================================================================
*/

// A single node can be deduplicated
template <>
class DedupAdapter<Nodep> {
 public:
  std::vector<Nodep> find_roots(const Nodep& n) const {
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
requires Dedupable<T>
class DedupAdapter<std::vector<T>> {
  DedupAdapter<T> t_helper{};

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
requires Dedupable<T>
class DedupAdapter<std::list<T>> {
  DedupAdapter<T> t_helper{};

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
requires Dedupable<T> && Dedupable<U>
class DedupAdapter<std::pair<T, U>> {
  DedupAdapter<T> t_helper{};
  DedupAdapter<U> u_helper{};

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
class DedupAdapter<double> {
 public:
  std::vector<Nodep> find_roots(const double&) const {
    return {};
  }
  double rewrite(const double& root, std::unordered_map<Nodep, Nodep>) const {
    return root;
  }
};

// A Real (wrapper around a double) can be deduplicated (no action)
template <>
class DedupAdapter<Real> {
 public:
  std::vector<Nodep> find_roots(const Real&) const {
    return {};
  }
  // rewrite the T, given a mapping from each node to its replacement.
  Real rewrite(const Real& t, const std::unordered_map<Nodep, Nodep>&) const {
    return t;
  }
};

} // namespace beanmachine::minibmg
