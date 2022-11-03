/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <sstream>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/irange.hpp>
#include <range/v3/action/push_back.hpp>
#include <range/v3/view/transform.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace beanmachine {
namespace util {

// Check if val is approximately zero
bool approx_zero(double val);

// sample with probability 1 / (1 + exp(-logodds))
bool sample_logodds(std::mt19937& gen, double logodds);

/*
Sample a boolean value given the log of the probability.
:param gen: random number generator
:param logprob: log of probability
:returns: true with probability exp(logprob), false otherwise
*/
bool sample_logprob(std::mt19937& gen, double logprob);

/*
A more intuitive name for :sample_logprob.
:param gen: random number generator
:param logprob: log of probability
:returns: true with probability exp(logprob), false otherwise
*/
bool flip_coin_with_log_prob(std::mt19937& gen, double logprob);

/*
Sample a value from a Beta distribution
:param gen: random number generator
:param a: shape parameter of Beta
:param b: shape parameter of Beta
*/
double sample_beta(std::mt19937& gen, double a, double b);

// compute  1 / (1 + exp(-logodds))
double logistic(double logodds);

/*
Compute the cumulative of the standard Normal upto x.
:param: x
:returns: N(0,1).cdf(x)
*/
double Phi(double x);

/*
Compute the cumulative of the standard Normal upto x.
See https://core.ac.uk/download/pdf/41787448.pdf
:param: x
:returns: N(0,1).cdf(x)  (approximately)
*/
double Phi_approx(double x);

/*
Inverse of the approximate Phi function.
:param z:
:returns x: s.t. Phi_approx(x) == z
*/
double Phi_approx_inv(double z);

/*
Compute the percentiles of a vector of values.
:param values: the vector of values
:param percs: the desired percentiles
:returns: a vector of percentiles
*/
template <typename T>
std::vector<T> percentiles(
    const std::vector<T>& values,
    const std::vector<double>& percs) {
  // copy the values out before sorting them
  std::vector<T> copy(values.begin(), values.end());
  std::sort(copy.begin(), copy.end());
  std::vector<double> result;
  for (auto p : percs) {
    result.push_back(copy[int(copy.size() * p)]);
  }
  return result;
}

/*
Equivalent to log of sum of exponentiations of values,
but more numerically stable.
:param values: vector of log values
:returns: log sum exp of values
*/
double log_sum_exp(const std::vector<double>& values);
double log_sum_exp(double a, double b);

/*
  Given log potentials log pot_i
  where potentials pot_i are an unnormalized probability distribution,
  return the normalized probability distribution p_i.
  p_i = pot_i/Z
  where Z is the normalization constant sum_i exp(log pot_i).
*/
std::vector<double> probs_given_log_potentials(std::vector<double> log_pot);

struct BinaryLogSumExp {
  double operator()(double a, double b) const {
    return log_sum_exp(a, b);
  }
};

/*
Compute the polygamma function. Note n=0 is the digamma function.
:param n:
:returns: polygamma(n, x)
*/
double polygamma(int n, double x);

/*
Compute `log(1 + exp(x))` with numerical stability.
See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
:param x:
:returns: log(1 + exp(x))
*/
double log1pexp(double x);

Eigen::MatrixXd log1pexp(const Eigen::MatrixXd& x);

/*
Compute `log(1 - exp(x))` with numerical stability for negative x values.
See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
:param x:
:returns: log(1 - exp(x))
*/
double log1mexp(double x);

Eigen::MatrixXd log1mexp(const Eigen::MatrixXd& x);

template <typename T>
std::vector<T> make_reserved_vector(size_t n) {
  std::vector<T> result;
  result.reserve(n);
  return result;
}

// Given a non-empty range,
// returns (first, first) if all elements in range are equal to first,
// and (first, other) for other != first otherwise.
template <typename InputIterator>
auto all_equal_in_non_empty_range(InputIterator beg, InputIterator end) {
  auto first = *beg;
  for (; beg != end; beg++) {
    if (*beg != first) {
      return std::make_pair(first, *beg);
    }
  }
  return std::make_pair(first, first);
}

// Returns the unique element if all elements in range are equal,
// or throw the result of 'make_exception_if_empty()' if range is empty,
// or throw the result of 'make_exception_if_not_unique(e1, e2)'
// if range contains at least two distinct elements e1, e2.
template <typename Iterator, typename F1, typename F2>
auto get_unique_element_if_any_or_throw_exceptions(
    Iterator begin,
    Iterator end,
    F1 make_exception_if_empty,
    F2 make_exception_if_not_unique) {
  if (begin == end) {
    throw make_exception_if_empty();
  }

  auto pair = all_equal_in_non_empty_range(begin, end);

  if (pair.first == pair.second) {
    return pair.first;
  } else {
    throw make_exception_if_not_unique(pair.first, pair.second);
  }
}

// Dynamically casts elements in a vector<T2*> to
// a vector<T1*> where T1 is a subclass of T2.
template <typename T1, typename T2>
std::vector<T1*> vector_dynamic_cast(const std::vector<T2*>& t2s) {
  std::vector<T1*> result(t2s.size());
  for (size_t i = 0; i != t2s.size(); i++) {
    result[i] = dynamic_cast<T1*>(t2s[i]);
  }
  return result;
}

template <typename Iterator, typename Function>
auto map(Iterator b, Iterator e, Function f) {
  return std::make_pair(
      boost::make_transform_iterator(b, f),
      boost::make_transform_iterator(e, f));
}

template <typename Container, typename Function>
auto map(const Container& c, Function f) {
  return std::make_pair(
      boost::make_transform_iterator(c.begin(), f),
      boost::make_transform_iterator(c.end(), f));
}

template <typename Container, typename R, typename... Args>
auto map2vec(const Container& c, std::function<R(Args...)> f) {
  return std::vector<R>(
      boost::make_transform_iterator(c.begin(), f),
      boost::make_transform_iterator(c.end(), f));
}

template <typename Container, typename T>
std::vector<T> collect2vec(const Container& c, std::function<bool(T)> p) {
  std::vector<T> result;
  for (auto t : c) {
    if (p(t)) {
      result.push_back(t);
    }
  }
  return result;
}

template <typename Iterator>
auto sum(const std::pair<Iterator, Iterator>& range) {
  return std::accumulate(range.first, range.second, 0.0);
}

/*
Computes the log normal density for n idd samples.
Takes the sum of samples as well as the sum of sample squares,
as well as mu (mean) and sigma (standard deviation).
*/
inline double log_normal_density(
    double sum_x,
    double sum_xsq,
    double mu,
    double sigma,
    unsigned n) {
  static const double half_of_log_2_pi = 0.5 * std::log(2 * M_PI);
  return (-std::log(sigma) - half_of_log_2_pi) * n -
      0.5 * (sum_xsq - 2 * mu * sum_x + mu * mu * n) / (sigma * sigma);
}

/*
Computes the log normal density for a sample.
Takes the sampled value
as well as mu (mean) and sigma (standard deviation).
*/
inline double log_normal_density(double x, double mu, double sigma) {
  return log_normal_density(x, x * x, mu, sigma, 1);
}

/*
Computes the log poisson probability for a sample.
Takes the sampled value k
as well as lambda (rate).
*/
inline auto log_poisson_probability(unsigned k, double lambda) {
  return k * std::log(lambda) - lambda - std::lgamma(k + 1);
}

/*
 * Returns a runtime_error exception
 * indicating that the feature of given name is
 * unsupported.
 */
inline std::runtime_error unsupported(const char* name) {
  return std::runtime_error(std::string(name) + " is unsupported");
}

template <typename T>
void erase_position(std::vector<T>& vector, std::size_t index) {
  vector.erase(vector.begin() + index);
}

/*
 * An iterable over an enum class.
 * If you have an enum class Foo with first value Foo::FIRST
 * and last value Foo::LAST, use
 * using FooIterable = EnumClassIterable<Foo, Foo::FIRST, Foo::LAST>;
 * for (auto value : FooIterable()) {
 }
 * Code provided in https://stackoverflow.com/a/31836401/3358488
 */
template <typename C, C beginVal, C endVal>
class EnumClassIterable {
  using val_t = typename std::underlying_type<C>::type;
  int val;

 public:
  explicit EnumClassIterable(const C& f) : val(static_cast<val_t>(f)) {}
  EnumClassIterable() : val(static_cast<val_t>(beginVal)) {}
  EnumClassIterable operator++() {
    ++val;
    return *this;
  }
  C operator*() {
    return static_cast<C>(val);
  }
  EnumClassIterable begin() {
    return *this;
  } // default ctor is good
  EnumClassIterable end() {
    static const EnumClassIterable endIter =
        ++EnumClassIterable(endVal); // cache it
    return endIter;
  }
  bool operator!=(const EnumClassIterable& i) {
    return val != i.val;
  }
};

/*
 * Iterables over integer ranges.
 * Source: https://codereview.stackexchange.com/a/52217
 */
template <class Integer>
decltype(auto) range(Integer first, Integer last) {
  return boost::irange(first, last);
}

template <class Integer, class StepSize>
decltype(auto) range(Integer first, Integer last, StepSize step_size) {
  return boost::irange(first, last, step_size);
}

template <class Integer>
decltype(auto) range(Integer last) {
  return boost::irange(static_cast<Integer>(0), last);
}

template <class Integer>
decltype(auto) range(std::pair<Integer, Integer> begin_end) {
  return boost::irange(begin_end.first, begin_end.second);
}

template <typename T>
inline auto find_index(const std::vector<T>& vec, const T& element) {
  return std::find(vec.begin(), vec.end(), element) - vec.begin();
}

/*
A generic function for breadth-first search on a directed graph.
Parameters:
- start: a starting vertex
- get_children: a function from a vertex to a range-v3 range of its children
- prune: a function from a vector to booleans b1 and b2 where b1 indicates
  whether the search should be pruned at the node, and b2 indicates
  whether the pruning should include the node itself
  (that is, whether the action should *not* be applied to node)
- abort: a function from a vector to booleans b1 and b2 where b1 indicates
  whether the search should be aborted altogether when the node is reached,
  and b2 indicates whether the abort should include the node itself
  (that is, whether the action should *not* be applied to node before aborting)
- action: a function applied to each reached vertex.
Returns whether the search completed successfully (as opposed to aborted).
*/
template <typename Vertex, typename Range>
bool bfs(
    const Vertex& start,
    std::function<Range(const Vertex&)> get_children,
    std::function<std::pair<bool, bool>(const Vertex&)> prune,
    std::function<std::pair<bool, bool>(const Vertex&)> abort,
    std::function<void(const Vertex&)> action) {
  std::set<Vertex> visited;
  std::list<Vertex> queue({start});
  while (not queue.empty()) {
    auto vertex = queue.front();
    queue.pop_front();
    if (visited.contains(vertex)) {
      continue;
    }
    visited.insert(vertex);
    auto abort_and_abort_self = abort(vertex);
    if (abort_and_abort_self.first) {
      if (not abort_and_abort_self.second) {
        action(vertex);
      }
      return false;
    }
    auto prune_and_prune_self = prune(vertex);
    if (prune_and_prune_self.first) {
      if (not prune_and_prune_self.second) {
        action(vertex);
      }
    } else {
      action(vertex);
      ranges::actions::push_back(queue, get_children(vertex));
    }
  }
  return true;
}

template <typename T>
std::function<void(const T& t)> push_back_to(std::vector<T>& vec) {
  return [&](const T& t) { vec.push_back(t); };
}

template <typename Range>
std::string join(const Range& range, const char* sep = ", ") {
  std::ostringstream os;
  bool first = true;
  for (const auto& element : range) {
    if (not first) {
      os << sep;
    }
    os << element;
    first = false;
  }
  return os.str();
}

} // namespace util
} // namespace beanmachine
