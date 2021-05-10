# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Callable, List

import beanmachine.ppl.compiler.bmg_nodes as bn
from torch import Tensor


def _tensor_to_label(t: Tensor) -> str:
    length = len(t.shape)
    if length == 0:
        return str(t.item())
    comma = "," if length == 1 else ",\\n"
    return "[" + comma.join(_tensor_to_label(c) for c in t) + "]"


def _tensor_val(node: bn.ConstantTensorNode) -> str:
    return _tensor_to_label(node.value)


def _val(node: bn.ConstantNode) -> str:
    if isinstance(node.value, Tensor):
        return _tensor_to_label(node.value)
    return str(node.value)


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
    bn.LogSumExpVectorNode: "LogSumExp",
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
    bn.ToMatrixNode: "ToMatrix",
    bn.ToPositiveRealNode: "ToPosReal",
    bn.ToProbabilityNode: "ToProb",
    bn.ToRealNode: "ToReal",
    bn.UniformNode: "Uniform",
    bn.UntypedConstantNode: _val,
}

_none = []
_left_right = ["left", "right"]
_operand = ["operand"]
_probability = ["probability"]


def _numbers(n: int) -> List[str]:
    return [str(x) for x in range(n)]


def _numbered(node: bn.BMGNode) -> List[str]:
    return _numbers(len(node.inputs))


def _to_matrix(node: bn.BMGNode) -> List[str]:
    return ["rows", "columns"] + _numbers(len(node.inputs) - 2)


_edge_labels = {
    bn.AdditionNode: _left_right,
    bn.BernoulliLogitNode: _probability,
    bn.BernoulliNode: _probability,
    bn.BetaNode: ["alpha", "beta"],
    bn.BinomialNode: ["count", "probability"],
    bn.BinomialLogitNode: ["count", "probability"],
    bn.BooleanNode: _none,
    bn.CategoricalNode: _probability,
    bn.Chi2Node: ["df"],
    bn.ComplementNode: _operand,
    bn.ConstantBooleanMatrixNode: _none,
    bn.ConstantNaturalMatrixNode: _none,
    bn.ConstantNegativeRealMatrixNode: _none,
    bn.ConstantPositiveRealMatrixNode: _none,
    bn.ConstantProbabilityMatrixNode: _none,
    bn.ConstantRealMatrixNode: _none,
    bn.ConstantTensorNode: _none,
    bn.DirichletNode: ["concentration"],
    bn.DivisionNode: _left_right,
    bn.EqualNode: _left_right,
    bn.ExpM1Node: _operand,
    bn.ExpNode: _operand,
    bn.ExpProductFactorNode: _numbered,
    bn.FlatNode: _none,
    bn.GammaNode: ["concentration", "rate"],
    bn.GreaterThanEqualNode: _left_right,
    bn.GreaterThanNode: _left_right,
    bn.HalfCauchyNode: ["scale"],
    bn.IfThenElseNode: ["condition", "consequence", "alternative"],
    bn.IndexNode: _left_right,
    bn.IndexNodeDeprecated: _left_right,
    bn.LessThanEqualNode: _left_right,
    bn.LessThanNode: _left_right,
    bn.Log1mexpNode: _operand,
    bn.LogisticNode: _operand,
    bn.LogNode: _operand,
    bn.LogSumExpNode: _numbered,
    bn.LogSumExpVectorNode: _operand,
    bn.MapNode: _numbered,
    bn.MatrixMultiplicationNode: _left_right,
    bn.MultiAdditionNode: _numbered,
    bn.MultiplicationNode: _left_right,
    bn.NaturalNode: _none,
    bn.NegateNode: _operand,
    bn.NegativeRealNode: _none,
    bn.NormalNode: ["mu", "sigma"],
    bn.NotEqualNode: _left_right,
    bn.NotNode: _operand,
    bn.Observation: _operand,
    bn.PhiNode: _operand,
    bn.PositiveRealNode: _none,
    bn.PowerNode: _left_right,
    bn.ProbabilityNode: _none,
    bn.Query: ["operator"],
    bn.RealNode: _none,
    bn.SampleNode: _operand,
    bn.StudentTNode: ["df", "loc", "scale"],
    bn.TensorNode: _numbered,
    bn.ToMatrixNode: _to_matrix,
    bn.ToPositiveRealNode: _operand,
    bn.ToProbabilityNode: _operand,
    bn.ToRealNode: _operand,
    bn.UniformNode: ["low", "high"],
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


def get_edge_labels(node: bn.BMGNode) -> List[str]:
    t = type(node)
    if t not in _edge_labels:
        return ["UNKNOWN"] * len(node.inputs)
    labels = _edge_labels[t]
    if isinstance(labels, list):
        result = labels
    else:
        assert isinstance(labels, Callable)
        result = labels(node)
    assert isinstance(result, list) and len(result) == len(node.inputs)
    return result


def get_edge_label(node: bn.BMGNode, i: int) -> str:
    t = type(node)
    if t not in _edge_labels:
        return "UNKNOWN"
    labels = _edge_labels[t]
    if isinstance(labels, list):
        return labels[i]
    assert isinstance(labels, Callable)
    if labels is _numbered:
        return str(i)
    return labels(node)[i]
