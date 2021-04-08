# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Callable

import beanmachine.ppl.compiler.bmg_nodes as bn
from torch import Tensor


def _val(node: bn.ConstantNode) -> str:
    return str(node.value)


def _tensor_to_label(t: Tensor) -> str:
    length = len(t.shape)
    if length == 0 or length == 1:
        return bn.ConstantTensorNode._tensor_to_python(t)
    return "[" + ",\\n".join(_tensor_to_label(c) for c in t) + "]"


def _tensor_val(node: bn.ConstantTensorNode) -> str:
    return _tensor_to_label(node.value)


_node_labels = {
    bn.AdditionNode: "+",
    bn.BernoulliLogitNode: "Bernoulli(logits)",
    bn.BernoulliNode: "Bernoulli",
    bn.BetaNode: "Beta",
    bn.BinomialNode: "Binomial",
    bn.BinomialLogitNode: "Binomial(logits)",
    bn.BooleanNode: _val,
    bn.CategoricalNode: lambda n: "Categorical(logits)"
    if n.is_logits
    else "Categorical",
    bn.Chi2Node: "Chi2",
    bn.ComplementNode: "complement",
    bn.ConstantBooleanMatrixNode: _tensor_val,
    bn.ConstantNaturalMatrixNode: _tensor_val,
    bn.ConstantNegativeRealMatrixNode: _tensor_val,
    bn.ConstantPositiveRealMatrixNode: _tensor_val,
    bn.ConstantProbabilityMatrixNode: _tensor_val,
    bn.ConstantRealMatrixNode: _tensor_val,
    bn.ConstantTensorNode: _tensor_val,
    bn.DirichletNode: "Dirichlet",
    bn.DivisionNode: "/",
    bn.EqualNode: "==",
    bn.ExpM1Node: "ExpM1",
    bn.ExpNode: "Exp",
    bn.ExpProductFactorNode: "ExpProduct",
    bn.FlatNode: "Flat",
    bn.GammaNode: "Gamma",
    bn.GreaterThanEqualNode: ">=",
    bn.GreaterThanNode: ">",
    bn.HalfCauchyNode: "HalfCauchy",
    bn.IfThenElseNode: "if",
    bn.IndexNode: "index",
    bn.IndexNodeDeprecated: "index",
    bn.LessThanEqualNode: "<=",
    bn.LessThanNode: "<",
    bn.Log1mexpNode: "Log1mexp",
    bn.LogisticNode: "Logistic",
    bn.LogNode: "Log",
    bn.LogSumExpNode: "LogSumExp",
    bn.MapNode: "map",
    bn.MatrixMultiplicationNode: "*",
    bn.MultiAdditionNode: "+",
    bn.MultiplicationNode: "*",
    bn.NaturalNode: _val,
    bn.NegateNode: "-",
    bn.NegativeRealNode: _val,
    bn.NormalNode: "Normal",
    bn.NotEqualNode: "!=",
    bn.NotNode: "not",
    bn.Observation: lambda n: f"Observation {str(n.value)}",
    bn.PhiNode: "Phi",
    bn.PositiveRealNode: _val,
    bn.PowerNode: "**",
    bn.ProbabilityNode: _val,
    bn.Query: "Query",
    bn.RealNode: _val,
    bn.SampleNode: "Sample",
    bn.StudentTNode: "StudentT",
    bn.TensorNode: "Tensor",
    bn.ToPositiveRealNode: "ToPosReal",
    bn.ToProbabilityNode: "ToProb",
    bn.ToRealNode: "ToReal",
    bn.UniformNode: "Uniform",
}


def get_node_label(node: bn.BMGNode) -> str:
    t = type(node)
    if t not in _node_labels:
        return "UNKNOWN"
    label = _node_labels[t]
    if isinstance(label, str):
        return label
    assert isinstance(label, Callable)
    return label(node)
