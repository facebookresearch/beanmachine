// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

void to_real(graph::Node* node);
void to_tensor(graph::Node* node);
void negate(graph::Node* node);
void exp(graph::Node* node);
void expm1(graph::Node* node);

} // namespace oper
} // namespace beanmachine
