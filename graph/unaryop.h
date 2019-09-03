// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "graph.h"

namespace beanmachine {
namespace oper {

void to_real(graph::Node* node);
void negate(graph::Node* node);
void exp(graph::Node* node);

} // namespace oper
} // namespace beanmachine
