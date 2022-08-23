/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <beanmachine/graph/graph.h>
#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/linalgop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"

bool ::beanmachine::oper::OperatorFactory::factories_are_registered =

    // control flow op
    OperatorFactory::register_op(
        graph::OperatorType::IF_THEN_ELSE,
        &(IfThenElse::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::CHOICE,
        &(Choice::new_op)) &&

    OperatorFactory::register_op(graph::OperatorType::ADD, &(Add::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::MULTIPLY,
        &(Multiply::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::LOGSUMEXP,
        &(LogSumExp::new_op)) &&

    OperatorFactory::register_op(graph::OperatorType::POW, &(Pow::new_op)) &&

    // stochastic op
    OperatorFactory::register_op(
        graph::OperatorType::SAMPLE,
        &(Sample::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::IID_SAMPLE,
        &(IIdSample::new_op)) &&

    // unary op
    OperatorFactory::register_op(
        graph::OperatorType::COMPLEMENT,
        &(Complement::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_INT,
        &(ToInt::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_REAL,
        &(ToReal::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_REAL_MATRIX,
        &(ToRealMatrix::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_POS_REAL,
        &(ToPosReal::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_POS_REAL_MATRIX,
        &(ToPosRealMatrix::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_PROBABILITY,
        &(ToProbability::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_NEG_REAL,
        &(ToNegReal::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::NEGATE,
        &(Negate::new_op)) &&

    OperatorFactory::register_op(graph::OperatorType::EXP, &(Exp::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::EXPM1,
        &(ExpM1::new_op)) &&

    OperatorFactory::register_op(graph::OperatorType::PHI, &(Phi::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::LOGISTIC,
        &(Logistic::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::LOG1PEXP,
        &(Log1pExp::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::LOG1MEXP,
        &(Log1mExp::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::LOGSUMEXP_VECTOR,
        &(LogSumExpVector::new_op)) &&

    OperatorFactory::register_op(graph::OperatorType::LOG, &(Log::new_op)) &&

    // linear algebra op
    OperatorFactory::register_op(
        graph::OperatorType::TRANSPOSE,
        &(Transpose::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::MATRIX_MULTIPLY,
        &(MatrixMultiply::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::MATRIX_SCALE,
        &(MatrixScale::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::ELEMENTWISE_MULTIPLY,
        &(ElementwiseMultiply::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::MATRIX_ADD,
        &(MatrixAdd::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::BROADCAST_ADD,
        &(BroadcastAdd::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::MATRIX_EXP,
        &(MatrixExp::new_op)) &&

    // matrix index
    OperatorFactory::register_op(
        graph::OperatorType::INDEX,
        &(Index::new_op)) &&

    // column index
    OperatorFactory::register_op(
        graph::OperatorType::COLUMN_INDEX,
        &(ColumnIndex::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::TO_MATRIX,
        &(ToMatrix::new_op)) &&

    // Cholesky decomposition
    OperatorFactory::register_op(
        graph::OperatorType::CHOLESKY,
        &(Cholesky::new_op)) &&

    // logProb function
    OperatorFactory::register_op(
        graph::OperatorType::LOG_PROB,
        &(LogProb::new_op)) &&

    OperatorFactory::register_op(
        graph::OperatorType::MATRIX_SUM,
        &(MatrixSum::new_op));
