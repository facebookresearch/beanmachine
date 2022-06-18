/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "beanmachine/minibmg/container.h"

using namespace ::beanmachine::minibmg;

// A container type
class Graph : public Container {};

// For testing purposes, we use a value that contains a unique
// sequence number for each instance created.
class SomeValue {
 public:
  int sequence = next_sequence++;
  static int next_sequence;
};

int SomeValue::next_sequence = 0;

// A Property, which associates a value with each container.
struct MyProperty : public Property<MyProperty, Graph, SomeValue> {
  SomeValue* create(Graph&) const override {
    // Code here to compute the value for the graph
    return new SomeValue();
  }
};

TEST(testcontainer, idempotence) {
  // reset state modified by the test.
  SomeValue::next_sequence = 0;

  // create two new containers
  Graph g1;
  Graph g2;

  // get the corresponding value for each.
  auto v2 = MyProperty::get(g2);
  auto v1 = MyProperty::get(g1);

  ASSERT_NE(v1, nullptr);
  ASSERT_NE(v2, nullptr);

  // assert that the values were created in the order accessed.
  ASSERT_EQ(v1->sequence, 1);
  ASSERT_EQ(v2->sequence, 0);

  // assert that the value is created only once.
  ASSERT_EQ(MyProperty::get(g1)->sequence, 1);
  ASSERT_EQ(MyProperty::get(g2)->sequence, 0);
}

class Value1 : public SomeValue {};
struct Property1 : public Property<Property1, Graph, Value1> {
  Value1* create(Graph& g) const override;
};

class Value2 : public SomeValue {};
struct Property2 : public Property<Property2, Graph, Value2> {
  Value2* create(Graph& g) const override;
};

Value1* Property1::create(Graph& g) const {
  // value1 needs value2
  (void)Property2::get(g);
  return new Value1();
}

Value2* Property2::create(Graph&) const {
  // value2 does not need value1
  return new Value2();
}

TEST(testcontainer, ordering) {
  // reset state modified by the test.
  SomeValue::next_sequence = 0;

  // values are created on demand.  As long as there is no dependence
  // cycle between value creators, they can reference each other.
  // TODO: we could detect cycles and report them (and test that we do).
  Graph g1;
  auto v1 = Property1::get(g1);
  auto v2 = Property2::get(g1);
  ASSERT_EQ(v1->sequence, 1);
  ASSERT_EQ(v2->sequence, 0);

  Graph g2;
  auto v3 = Property2::get(g2);
  auto v4 = Property1::get(g2);
  ASSERT_EQ(v3->sequence, 2);
  ASSERT_EQ(v4->sequence, 3);
}
