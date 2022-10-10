/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_properties/container.h"

namespace beanmachine::minibmg {

Container::~Container() {
  // delete all the values contained.
  for (auto i = _container_map.begin(), end = _container_map.end(); i != end;
       ++i) {
    const _BaseProperty* const property = i->first;
    const void* value = i->second;
    property->_delete_value(value);
  }
}

} // namespace beanmachine::minibmg
