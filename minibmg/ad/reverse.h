/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/topological.h"

namespace beanmachine::minibmg {

template <class Underlying>
requires Number<Underlying>
class ReverseBody;

/*
 * An implementation of numbers offering reverse-mode differentiation.
 * Can be used to automatically compute derivatives of more complex functions
 * by using the overloaded operators for computing new values and their
 * derivatives.  Implements the Number concept.
 */
template <class Underlying>
requires Number<Underlying>
class Reverse {
 public:
  using Bodyp = std::shared_ptr<ReverseBody<Underlying>>;
  Bodyp ptr;

  Reverse();
  /* implicit */ Reverse(double primal);
  /* implicit */ Reverse(Underlying primal);
  /* implicit */ Reverse(Bodyp body);
  Reverse(const Reverse<Underlying>& other);
  virtual ~Reverse();
  Reverse<Underlying>& operator=(const Reverse<Underlying>& other);
  double as_double() const;
  void reverse(double initial_gradient);
};

template <class Underlying>
requires Number<Underlying>
class ReverseBody {
 public:
  Underlying primal;
  std::list<Reverse<Underlying>> inputs;
  Underlying gradient = 0;

  ReverseBody(double primal);
  ReverseBody(Underlying primal);
  ReverseBody(Underlying primal, const std::list<Reverse<Underlying>>& inputs);
  ReverseBody(const ReverseBody<Underlying>& other);
  ReverseBody<Underlying>& operator=(const ReverseBody<Underlying>& other);
  virtual ~ReverseBody() {}
  virtual void backprop();
};

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>::Reverse(double primal)
    : ptr{std::make_shared<ReverseBody<Underlying>>(primal)} {}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>::Reverse(Underlying primal)
    : ptr{std::make_shared<ReverseBody<Underlying>>(primal)} {}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>::Reverse()
    : ptr{std::make_shared<ReverseBody<Underlying>>(0)} {}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>::~Reverse() {}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>::Reverse(
    std::shared_ptr<ReverseBody<Underlying>> body)
    : ptr{body} {}

// This implementation of ReverseBody<Underlying> permits the precomputation of
// the local derivative, and simply multiplies during the reverse phase to
// implement the chain rule.  Although not necessarily storage efficient for
// tensors, for scalars it is as good as anything else.
template <class Underlying>
class PrecomputedGradients : public ReverseBody<Underlying> {
 public:
  const std::list<Underlying> computed_gradients;
  PrecomputedGradients(
      Underlying primal,
      const std::list<Reverse<Underlying>>& inputs,
      const std::list<Underlying>& computed_gradients)
      : ReverseBody<Underlying>{primal, inputs},
        computed_gradients{computed_gradients} {}
  void backprop() override {
    auto& /*std::list<Reverse<Underlying>>*/ t = this->inputs;
    typename std::list<Reverse<Underlying>>::iterator it1 = t.begin();
    typename std::list<Underlying>::const_iterator it2 =
        computed_gradients.begin();
    for (; it1 != t.end() && it2 != computed_gradients.end(); ++it1, ++it2) {
      auto& grad = *it2;
      auto& input_grad = it1->ptr->gradient;
      input_grad = input_grad + this->gradient * grad;
    }
  }
};

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>::Reverse(
    const Reverse<Underlying>& other)
    : ptr{other.ptr} {}

template <class Underlying>
requires Number<Underlying>
double Reverse<Underlying>::as_double() const {
  return ptr->as_double();
}

template <class Underlying>
requires Number<Underlying>
void Reverse<Underlying>::reverse(double initial_gradient) {
  // topologically sort the set of ReverseBody pointers reachable - these are
  // the nodes to which we need to backprop.
  std::list<Bodyp> roots = {ptr};
  std::function<std::vector<Bodyp>(const Bodyp&)> predecessors =
      [&](const Bodyp& ptr) -> std::vector<Bodyp> {
    std::vector<Bodyp> predecessors;
    for (auto& i : ptr->inputs) {
      predecessors.push_back(i.ptr);
    }
    return predecessors;
  };
  std::vector<Bodyp> sorted;
  if (!topological_sort<Bodyp>(roots, predecessors, sorted)) {
    throw std::logic_error("reverse nodes have a loop");
  }

  // clear their gradients
  for (auto p : sorted) {
    p->gradient = 0;
  }

  // set the initial gradient
  ptr->gradient = initial_gradient;

  // go through the list in order and call backprop() on each.
  for (auto p : sorted) {
    p->backprop();
  }
}

template <class Underlying>
requires Number<Underlying> ReverseBody<Underlying>::ReverseBody(double primal)
    : primal{primal} {}

template <class Underlying>
requires Number<Underlying> ReverseBody<Underlying>::ReverseBody(
    Underlying primal)
    : primal{primal} {}

template <class Underlying>
requires Number<Underlying> ReverseBody<Underlying>::ReverseBody(
    Underlying primal,
    const std::list<Reverse<Underlying>>& inputs)
    : primal{primal}, inputs{inputs} {}

template <class Underlying>
requires Number<Underlying>
void ReverseBody<Underlying>::backprop() {}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>
operator+(const Reverse<Underlying>& left, const Reverse<Underlying>& right) {
  Underlying new_primal = left.ptr->primal + right.ptr->primal;
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{left, right},
      std::list<Underlying>{1, 1})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>
operator-(const Reverse<Underlying>& left, const Reverse<Underlying>& right) {
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      left.ptr->primal - right.ptr->primal,
      std::list<Reverse<Underlying>>{left, right},
      std::list<Underlying>{1, -1})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>
operator-(const Reverse<Underlying>& x) {
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      -x.ptr->primal,
      std::list<Reverse<Underlying>>{x},
      std::list<Underlying>{-1})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>
operator*(const Reverse<Underlying>& left, const Reverse<Underlying>& right) {
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      left.ptr->primal * right.ptr->primal,
      std::list<Reverse<Underlying>>{left, right},
      std::list<Underlying>{right.ptr->primal, left.ptr->primal})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying>
operator/(const Reverse<Underlying>& left, const Reverse<Underlying>& right) {
  // a / b
  Underlying new_primal = left.ptr->primal / right.ptr->primal;

  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{left, right},
      std::list<Underlying>{
          1 / right.ptr->primal, -new_primal / right.ptr->primal})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> pow(
    const Reverse<Underlying>& base,
    const Reverse<Underlying>& exponent) {
  double power;
  // shortcut some cases.
  if (is_constant(exponent, power)) {
    if (power == 0)
      return 1;
    if (power == 1)
      return base;
  }

  Underlying new_primal = pow(base.ptr->primal, exponent.ptr->primal);

  // Doing these gradients symbolically is too hard for my small brain.  So
  // we'll use Num2 to get the answer.  Eventually we should come back and
  // improve this.
  Underlying grad1 = pow(Num2<Underlying>{base.ptr->primal, 1},
                         Num2<Underlying>{exponent.ptr->primal})
                         .derivative1;
  Underlying grad2 = pow(Num2<Underlying>{base.ptr->primal},
                         Num2<Underlying>{exponent.ptr->primal, 1})
                         .derivative1;
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{base, exponent},
      std::list<Underlying>{grad1, grad2})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> exp(
    const Reverse<Underlying>& x) {
  Underlying new_primal = exp(x.ptr->primal);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{x},
      std::list<Underlying>{new_primal})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> log(
    const Reverse<Underlying>& x) {
  Underlying new_primal = log(x.ptr->primal);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{x},
      std::list<Underlying>{1 / x.ptr->primal})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> atan(
    const Reverse<Underlying>& x) {
  Underlying new_primal = atan(x.ptr->primal);
  Underlying new_derivative1 = 1 / (x.ptr->primal * x.ptr->primal + 1.0f);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{x},
      std::list<Underlying>{new_derivative1})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> lgamma(
    const Reverse<Underlying>& x) {
  Underlying new_primal = lgamma(x.ptr->primal);
  Underlying new_derivative1 = polygamma(0, x.ptr->primal);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{x},
      std::list<Underlying>{new_derivative1})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> polygamma(
    int n,
    const Reverse<Underlying>& x) {
  Underlying new_primal = polygamma(n, x.ptr->primal);
  Underlying new_derivative1 = polygamma(n + 1, x.ptr->primal);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{x},
      std::list<Underlying>{new_derivative1})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> if_equal(
    const Reverse<Underlying>& value,
    const Reverse<Underlying>& comparand,
    const Reverse<Underlying>& when_equal,
    const Reverse<Underlying>& when_not_equal) {
  // Note: we discard and ignore left.derivative1 and
  // comparand->derivative1
  Underlying new_primal = if_equal(
      value.ptr->primal,
      comparand.ptr->primal,
      when_equal.ptr->primal,
      when_not_equal.ptr->primal);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{when_equal, when_not_equal},
      std::list<Underlying>{
          if_equal(value.ptr->primal, comparand.ptr->primal, 1, 0),
          if_equal(value.ptr->primal, comparand.ptr->primal, 0, 1)})};
}

template <class Underlying>
requires Number<Underlying> Reverse<Underlying> if_less(
    const Reverse<Underlying>& value,
    const Reverse<Underlying>& comparand,
    const Reverse<Underlying>& when_less,
    const Reverse<Underlying>& when_not_less) {
  // Note: we discard and ignore left.derivative1 and
  // comparand->derivative1
  Underlying new_primal = if_less(
      value.ptr->primal,
      comparand.ptr->primal,
      when_less.ptr->primal,
      when_not_less.ptr->primal);
  return Reverse<Underlying>{std::make_shared<PrecomputedGradients<Underlying>>(
      new_primal,
      std::list<Reverse<Underlying>>{when_less, when_not_less},
      std::list<Underlying>{
          if_less(value.ptr->primal, comparand.ptr->primal, 1, 0),
          if_less(value.ptr->primal, comparand.ptr->primal, 0, 1)})};
}

template <class Underlying>
requires Number<Underlying>
bool is_constant(const Reverse<Underlying>& x, double& value) {
  if (typeid(x.ptr) == typeid(ReverseBody<Underlying>)) {
    // this is the only class in the hierarchy that never backpropagates.
    return is_constant(x.ptr->primal, value);
  }
  return false;
}

template <class Underlying>
requires Number<Underlying>
bool is_constant(const Reverse<Underlying>& x, const double& value) {
  double v;
  return is_constant(x, v) && value == v;
}

template <class Underlying>
requires Number<Underlying> std::string to_string(
    const Reverse<Underlying>& x) {
  return fmt::format("[back primal={0}]", to_string(x.ptr->primal));
}

static_assert(Number<Reverse<Real>>);

} // namespace beanmachine::minibmg
