// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

void multiply(graph::Node* node);
void add(graph::Node* node);
void logsumexp(graph::Node* node);
void pow(graph::Node* node);

} // namespace oper
} // namespace beanmachine
