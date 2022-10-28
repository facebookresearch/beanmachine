/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// In order to deduplicate data in a given data structure (or use the local
// optimizer, or other node rewriting APIs), the programmer must specialize this
// template class to (1) locate the roots contained in that data structure, and
// (2) write a replacement data structure in which nodes (values of type Nodep)
// have been deduplicated.  Those methods should be provided by the programmer
// in the specialization and have the signatures of the two methods shown in the
// body of this class.  They are commented out here because we don't have a way
// of doing it for all types, and we don't want this unspecialized class to
// satisfy the `Rewritable` concept.  We provide a number of specializations for
// data structures likely to be needed.  `T` here is the type of the data
// structure for which nodes contained in it are to be deduplicated.
//
// The idea of organizing the code this way, with a type that the caller may
// specialize, is something I learned from the C++ standard template library.
// See std::hash<Key> and std::equal_to<Key> and their use in
// std::unordered_map.
template <class T>
class NodeRewriteAdapter {
 public:
  NodeRewriteAdapter() = delete;

  // To implement the `Rewritable` concept, you can specialize this template
  // class and provide methods with the signatures shown below.

  // locate all of the roots.
  // std::vector<Nodep> find_roots(const T&) const;

  // rewrite the T, given a mapping from each node to its replacement.
  // T rewrite(const T&, const std::unordered_map<Nodep, Nodep>&) const;
};

// A cancept asserting that the type `T` is a container that may contain nodes
// (Nodep) that can be rewritten with the help of RewriteAdapter.  Rewriters
// include `dedup` (which performs common subexpression elimination) and `opt`
// (which performs a set of local optimizations).  An adapter has operations to
// (1) find the nodes appearing in the data structure, and (2) write a new
// version of the data structure, replacing nodes as indicated by a map.
template <class T, class RewriteAdapter = NodeRewriteAdapter<T>>
concept Rewritable = requires(
    const T& t,
    const RewriteAdapter& a,
    const std::unordered_map<Nodep, Nodep>& map) {
  { a.find_roots(t) } -> std::convertible_to<std::vector<Nodep>>;
  { a.rewrite(t, map) } -> std::same_as<T>;
  // TODO: require that RewriteAdapter is default-constructible, as its instance
  // methods would not be usable without an instance.  Once we do that we can
  // uncomment the two methods in NodeRewriteAdapter as a guide for programmers
  // specializing it, as the lack of a default constructor would be enough for
  // it to fail to satisfy the `Rewritable` concept.
};

// A single node can be deduplicated
template <>
class NodeRewriteAdapter<Nodep> {
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
static_assert(Rewritable<Nodep>);

// A vector can be deduplicated.
template <class T>
requires Rewritable<T>
class NodeRewriteAdapter<std::vector<T>> {
  NodeRewriteAdapter<T> t_helper{};

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
static_assert(Rewritable<std::vector<Nodep>>);

// A pair can be deduplicated
template <class T, class U>
requires Rewritable<T> && Rewritable<U>
class NodeRewriteAdapter<std::pair<T, U>> {
  NodeRewriteAdapter<T> t_helper{};
  NodeRewriteAdapter<U> u_helper{};

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
static_assert(Rewritable<std::pair<Nodep, Nodep>>);

// A double can be deduplicated (no action)
template <>
class NodeRewriteAdapter<double> {
 public:
  std::vector<Nodep> find_roots(const double&) const {
    return {};
  }
  double rewrite(const double& root, std::unordered_map<Nodep, Nodep>) const {
    return root;
  }
};
static_assert(Rewritable<double>);

// A Real (wrapper around a double) can be deduplicated (no action)
template <>
class NodeRewriteAdapter<Real> {
 public:
  std::vector<Nodep> find_roots(const Real&) const {
    return {};
  }
  // rewrite the T, given a mapping from each node to its replacement.
  Real rewrite(const Real& t, const std::unordered_map<Nodep, Nodep>&) const {
    return t;
  }
};
static_assert(Rewritable<std::vector<Real>>);

} // namespace beanmachine::minibmg
