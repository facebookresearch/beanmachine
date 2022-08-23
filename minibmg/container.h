/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace beanmachine::minibmg {

class _BaseProperty;

/*
 * A container that can hold client-defined values, each identified by a
 * Property. Extend this class to make a data structure extensible by its
 * clients to contain client-specific data.
 */
class Container {
 public:
  Container() {}
  virtual ~Container();

  std::unordered_map<const _BaseProperty*, void*> _container_map;
};

/*
 * A base class used for properties.  Part of the internal implementation of
 * Container and Property.
 */
class _BaseProperty {
 public:
  _BaseProperty() {}

  virtual void _delete_value(void* valuep) const = 0;

  virtual ~_BaseProperty() {}
};

/*
 * A property, which identifies a pseudo-field of the given container type.
 * Each property type represents one value associated with each instance
 * of the container.  Derive from this class to define a new property which
 * identifies the field.  The full usage pattern is as follows:
 *
 * // Define a data type that will be extensible with client-defined data
 *
 * class MyContainerType : Container ...
 *
 * // Define a property for data associated with each instance of a container
 * // along with a method for creating new data the first time it is accessed.
 *
 * struct MyProperty : Property<MyProperty, MyContainerType, TypeOfData> {
 *  TypeOfData* create(MyContainerType& c) const override { ... }
 * }
 *
 * // Given a container, get the value associated with that property
 *
 * MyContainerType my = ...
 * TypeOfData* pseudoFieldValue = MyProperty::get(my);
 */
template <class DerivedPropertyType, class ContainerType, class ValueType>
class Property : public _BaseProperty {
 public:
  static ValueType* get(const ContainerType& container);
  void _delete_value(void* valuep) const override {
    auto vp = (ValueType*)valuep;
    delete vp;
  }

  virtual ~Property() override {}

  virtual ValueType* create(const ContainerType& container) const = 0;

 protected:
  constexpr Property() {
    static_assert(
        std::is_base_of<
            Property<DerivedPropertyType, ContainerType, ValueType>,
            DerivedPropertyType>::value,
        "DerivedPropertyType (first) type parameter of Property<*,,> must be the class that uses this as a base class");
    static_assert(
        std::is_base_of<Container, ContainerType>::value,
        "ContainerType (second) type parameter of Property<,,*> must derive from Container");
  }
};

template <class DerivedPropertyType, class ContainerType, class ValueType>
ValueType* Property<DerivedPropertyType, ContainerType, ValueType>::get(
    const ContainerType& container) {
  // We enforce the singleton pattern by creating a single instance of the
  // property here. That simplifies the usage pattern, becuse the client merely
  // needs to declare a property, and then can use it to fetch values from a
  // container, as shown in the usage pattern comment on class Property.
  static const DerivedPropertyType singleton_property;
  const DerivedPropertyType* property = &singleton_property;
  auto& map = container._container_map;
  auto found = map.find(property);
  if (found != map.end()) {
    return (ValueType*)found->second;
  }
  auto result = property->create(container);
  // We cast away const to make the map mutable.  The API is still immutable, as
  // it is not possible for a client to observe a state transition (e.g. a
  // different state at two different times). That is because the client always
  // (only) sees the value in its constructed state.
  auto& mutable_map =
      *const_cast<std::unordered_map<const _BaseProperty*, void*>*>(&map);
  mutable_map[property] = result;
  return result;
}

} // namespace beanmachine::minibmg
