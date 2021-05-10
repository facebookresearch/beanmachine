# Copyright (c) Facebook, Inc. and its affiliates.
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
    bn.DirichletNode: (dt.DIRICHLET, None),
    bn.FlatNode: (dt.FLAT, AtomicType.PROBABILITY),
    bn.GammaNode: (dt.GAMMA, AtomicType.POS_REAL),
    bn.HalfCauchyNode: (dt.HALF_CAUCHY, AtomicType.POS_REAL),
    bn.NormalNode: (dt.NORMAL, AtomicType.REAL),
    bn.StudentTNode: (dt.STUDENT_T, AtomicType.REAL),
}


def dist_type(node: bn.DistributionNode) -> Tuple[dt, Any]:
    t = type(node)
    if t is bn.DirichletNode:
        element_type = ValueType(
            VariableType.COL_SIMPLEX_MATRIX,
            AtomicType.PROBABILITY,
            node._required_columns,
            1,
        )
        return dt.DIRICHLET, element_type
    return _dist_types[t]


_operator_types = {
    bn.AdditionNode: OperatorType.ADD,
    bn.ColumnIndexNode: OperatorType.COLUMN_INDEX,
    bn.ComplementNode: OperatorType.COMPLEMENT,
    bn.ExpM1Node: OperatorType.EXPM1,
    bn.ExpNode: OperatorType.EXP,
    bn.IfThenElseNode: OperatorType.IF_THEN_ELSE,
    bn.IndexNode: OperatorType.INDEX,
    bn.Log1mexpNode: OperatorType.LOG1MEXP,
    bn.LogNode: OperatorType.LOG,
    bn.LogisticNode: OperatorType.LOGISTIC,
    bn.LogSumExpNode: OperatorType.LOGSUMEXP,
    bn.LogSumExpVectorNode: OperatorType.LOGSUMEXP_VECTOR,
    bn.MultiAdditionNode: OperatorType.ADD,
    bn.MultiplicationNode: OperatorType.MULTIPLY,
    bn.NegateNode: OperatorType.NEGATE,
    bn.PhiNode: OperatorType.PHI,
    bn.PowerNode: OperatorType.POW,
    bn.SampleNode: OperatorType.SAMPLE,
    bn.ToMatrixNode: OperatorType.TO_MATRIX,
    bn.ToRealNode: OperatorType.TO_REAL,
    bn.ToPositiveRealNode: OperatorType.TO_POS_REAL,
    bn.ToProbabilityNode: OperatorType.TO_PROBABILITY,
}


def operator_type(node: bn.OperatorNode) -> OperatorType:
    return _operator_types[type(node)]


def is_supported_by_bmg(node: bn.BMGNode) -> bool:
    t = type(node)
    return (
        isinstance(node, bn.ConstantNode)
        or t is bn.Observation
        or t is bn.Query
        or t in _operator_types
        or t in _dist_types
        or t in _factor_types
    )
