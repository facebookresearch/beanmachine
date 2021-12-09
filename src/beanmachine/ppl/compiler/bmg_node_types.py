# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this module.
# pyre-ignore-all-errors

from typing import Any, Tuple

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.graph import (
    AtomicType,
    DistributionType as dt,
    FactorType as ft,
    OperatorType,
    ValueType,
    VariableType,
)
from beanmachine.ppl.compiler.bmg_types import SimplexMatrix
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


_factor_types = {
    bn.ExpProductFactorNode: ft.EXP_PRODUCT,
}


def factor_type(node: bn.FactorNode) -> ft:
    return _factor_types[type(node)]


_dist_types = {
    bn.BernoulliLogitNode: (dt.BERNOULLI_LOGIT, AtomicType.BOOLEAN),
    bn.BernoulliNode: (dt.BERNOULLI, AtomicType.BOOLEAN),
    bn.BetaNode: (dt.BETA, AtomicType.PROBABILITY),
    bn.BinomialNode: (dt.BINOMIAL, AtomicType.NATURAL),
    bn.CategoricalNode: (dt.CATEGORICAL, AtomicType.NATURAL),
    bn.DirichletNode: (dt.DIRICHLET, None),
    bn.FlatNode: (dt.FLAT, AtomicType.PROBABILITY),
    bn.GammaNode: (dt.GAMMA, AtomicType.POS_REAL),
    bn.HalfCauchyNode: (dt.HALF_CAUCHY, AtomicType.POS_REAL),
    bn.NormalNode: (dt.NORMAL, AtomicType.REAL),
    bn.HalfNormalNode: (dt.HALF_NORMAL, AtomicType.POS_REAL),
    bn.StudentTNode: (dt.STUDENT_T, AtomicType.REAL),
}


def dist_type(node: bn.DistributionNode) -> Tuple[dt, Any]:
    t = type(node)
    if t is bn.DirichletNode:
        simplex = LatticeTyper()[node]
        assert isinstance(simplex, SimplexMatrix)
        element_type = ValueType(
            VariableType.COL_SIMPLEX_MATRIX,
            AtomicType.PROBABILITY,
            simplex.rows,
            simplex.columns,
        )
        return dt.DIRICHLET, element_type
    return _dist_types[t]


_operator_types = {
    bn.AdditionNode: OperatorType.ADD,
    bn.ChoiceNode: OperatorType.CHOICE,
    bn.ColumnIndexNode: OperatorType.COLUMN_INDEX,
    bn.ComplementNode: OperatorType.COMPLEMENT,
    bn.ExpM1Node: OperatorType.EXPM1,
    bn.ExpNode: OperatorType.EXP,
    bn.IfThenElseNode: OperatorType.IF_THEN_ELSE,
    bn.Log1mexpNode: OperatorType.LOG1MEXP,
    bn.LogNode: OperatorType.LOG,
    bn.LogisticNode: OperatorType.LOGISTIC,
    bn.LogSumExpNode: OperatorType.LOGSUMEXP,
    bn.LogSumExpVectorNode: OperatorType.LOGSUMEXP_VECTOR,
    bn.MultiplicationNode: OperatorType.MULTIPLY,
    bn.MatrixScaleNode: OperatorType.MATRIX_SCALE,
    bn.NegateNode: OperatorType.NEGATE,
    bn.PhiNode: OperatorType.PHI,
    bn.PowerNode: OperatorType.POW,
    bn.SampleNode: OperatorType.SAMPLE,
    bn.ToIntNode: OperatorType.TO_INT,
    bn.ToMatrixNode: OperatorType.TO_MATRIX,
    bn.ToNegativeRealNode: OperatorType.TO_NEG_REAL,
    bn.ToRealMatrixNode: OperatorType.TO_REAL_MATRIX,
    bn.ToRealNode: OperatorType.TO_REAL,
    bn.ToPositiveRealMatrixNode: OperatorType.TO_POS_REAL_MATRIX,
    bn.ToPositiveRealNode: OperatorType.TO_POS_REAL,
    bn.ToProbabilityNode: OperatorType.TO_PROBABILITY,
    bn.VectorIndexNode: OperatorType.INDEX,
}

_constant_value_types = {
    bn.BooleanNode,
    bn.NaturalNode,
    bn.NegativeRealNode,
    bn.PositiveRealNode,
    bn.ProbabilityNode,
    bn.RealNode,
}

_constant_matrix_types = {
    bn.ConstantBooleanMatrixNode,
    bn.ConstantNaturalMatrixNode,
    bn.ConstantNegativeRealMatrixNode,
    bn.ConstantPositiveRealMatrixNode,
    bn.ConstantProbabilityMatrixNode,
    bn.ConstantRealMatrixNode,
    bn.ConstantSimplexMatrixNode,
}


def operator_type(node: bn.OperatorNode) -> OperatorType:
    return _operator_types[type(node)]


def is_supported_by_bmg(node: bn.BMGNode) -> bool:
    t = type(node)
    return (
        t is bn.Observation
        or t is bn.Query
        or t in _constant_matrix_types
        or t in _constant_value_types
        or t in _operator_types
        or t in _dist_types
        or t in _factor_types
    )
