/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/linalgop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"

namespace beanmachine {
namespace oper {

// control flow op
bool IfThenElse::is_registered = OperatorFactory::register_op(
    graph::OperatorType::IF_THEN_ELSE,
    &(IfThenElse::new_op));

bool Choice::is_registered = OperatorFactory::register_op(
    graph::OperatorType::CHOICE,
    &(Choice::new_op));

// multiary op
bool Add::is_registered =
    OperatorFactory::register_op(graph::OperatorType::ADD, &(Add::new_op));

bool Multiply::is_registered = OperatorFactory::register_op(
    graph::OperatorType::MULTIPLY,
    &(Multiply::new_op));

bool LogSumExp::is_registered = OperatorFactory::register_op(
    graph::OperatorType::LOGSUMEXP,
    &(LogSumExp::new_op));

bool Pow::is_registered =
    OperatorFactory::register_op(graph::OperatorType::POW, &(Pow::new_op));

// stochastic op
bool Sample::is_registered = OperatorFactory::register_op(
    graph::OperatorType::SAMPLE,
    &(Sample::new_op));

bool IIdSample::is_registered = OperatorFactory::register_op(
    graph::OperatorType::IID_SAMPLE,
    &(IIdSample::new_op));

// unary op
bool Complement::is_registered = OperatorFactory::register_op(
    graph::OperatorType::COMPLEMENT,
    &(Complement::new_op));

bool ToInt::is_registered =
    OperatorFactory::register_op(graph::OperatorType::TO_INT, &(ToInt::new_op));

bool ToReal::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_REAL,
    &(ToReal::new_op));

bool ToRealMatrix::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_REAL_MATRIX,
    &(ToRealMatrix::new_op));

bool ToPosReal::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_POS_REAL,
    &(ToPosReal::new_op));

bool ToPosRealMatrix::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_POS_REAL_MATRIX,
    &(ToPosRealMatrix::new_op));

bool ToProbability::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_PROBABILITY,
    &(ToProbability::new_op));

bool ToNegReal::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_NEG_REAL,
    &(ToNegReal::new_op));

bool Negate::is_registered = OperatorFactory::register_op(
    graph::OperatorType::NEGATE,
    &(Negate::new_op));

bool Exp::is_registered =
    OperatorFactory::register_op(graph::OperatorType::EXP, &(Exp::new_op));

bool ExpM1::is_registered =
    OperatorFactory::register_op(graph::OperatorType::EXPM1, &(ExpM1::new_op));

bool Phi::is_registered =
    OperatorFactory::register_op(graph::OperatorType::PHI, &(Phi::new_op));

bool Logistic::is_registered = OperatorFactory::register_op(
    graph::OperatorType::LOGISTIC,
    &(Logistic::new_op));

bool Log1pExp::is_registered = OperatorFactory::register_op(
    graph::OperatorType::LOG1PEXP,
    &(Log1pExp::new_op));

bool Log1mExp::is_registered = OperatorFactory::register_op(
    graph::OperatorType::LOG1MEXP,
    &(Log1mExp::new_op));

bool LogSumExpVector::is_registered = OperatorFactory::register_op(
    graph::OperatorType::LOGSUMEXP_VECTOR,
    &(LogSumExpVector::new_op));

bool Log::is_registered =
    OperatorFactory::register_op(graph::OperatorType::LOG, &(Log::new_op));

// linear algebra op
bool MatrixMultiply::is_registered = OperatorFactory::register_op(
    graph::OperatorType::MATRIX_MULTIPLY,
    &(MatrixMultiply::new_op));

bool MatrixScale::is_registered = OperatorFactory::register_op(
    graph::OperatorType::MATRIX_SCALE,
    &(MatrixScale::new_op));

bool BroadcastAdd::is_registered = OperatorFactory::register_op(
    graph::OperatorType::BROADCAST_ADD,
    &(BroadcastAdd::new_op));

// matrix index
bool Index::is_registered =
    OperatorFactory::register_op(graph::OperatorType::INDEX, &(Index::new_op));

// matrix index
bool ColumnIndex::is_registered = OperatorFactory::register_op(
    graph::OperatorType::COLUMN_INDEX,
    &(ColumnIndex::new_op));

// matrix index
bool ToMatrix::is_registered = OperatorFactory::register_op(
    graph::OperatorType::TO_MATRIX,
    &(ToMatrix::new_op));

} // namespace oper
} // namespace beanmachine
