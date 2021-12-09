# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    bn.BitAndNode: "&",
    bn.BitOrNode: "|",
    bn.BitXorNode: "^",
    bn.BooleanNode: _val,
    bn.CategoricalLogitNode: "Categorical(logits)",
    bn.CategoricalNode: "Categorical",
    bn.Chi2Node: "Chi2",
    bn.ChoiceNode: "Choice",
    bn.ColumnIndexNode: "ColumnIndex",
    bn.ComplementNode: "complement",
    bn.ConstantBooleanMatrixNode: _tensor_val,
    bn.ConstantNaturalMatrixNode: _tensor_val,
    bn.ConstantNegativeRealMatrixNode: _tensor_val,
    bn.ConstantPositiveRealMatrixNode: _tensor_val,
    bn.ConstantProbabilityMatrixNode: _tensor_val,
    bn.ConstantRealMatrixNode: _tensor_val,
    bn.ConstantSimplexMatrixNode: _tensor_val,
    bn.ConstantTensorNode: _tensor_val,
    bn.DirichletNode: "Dirichlet",
    bn.DivisionNode: "/",
    bn.EqualNode: "==",
    bn.ExpM1Node: "ExpM1",
    bn.ExpNode: "Exp",
    bn.ExpProductFactorNode: "ExpProduct",
    bn.FlatNode: "Flat",
    bn.FloorDivNode: "//",
    bn.GammaNode: "Gamma",
    bn.GreaterThanEqualNode: ">=",
    bn.GreaterThanNode: ">",
    bn.HalfCauchyNode: "HalfCauchy",
    bn.IfThenElseNode: "if",
    bn.InNode: "In",
    bn.IndexNode: "index",
    bn.InvertNode: "Invert",
    bn.IsNode: "Is",
    bn.IsNotNode: "IsNot",
    bn.ItemNode: "Item",
    bn.LessThanEqualNode: "<=",
    bn.LessThanNode: "<",
    bn.Log1mexpNode: "Log1mexp",
    bn.LogisticNode: "Logistic",
    bn.LogNode: "Log",
    bn.LogSumExpNode: "LogSumExp",
    bn.LogSumExpTorchNode: "LogSumExp",
    bn.LogSumExpVectorNode: "LogSumExp",
    bn.LShiftNode: "<<",
    bn.MatrixMultiplicationNode: "@",
    bn.MatrixScaleNode: "MatrixScale",
    bn.ModNode: "%",
    bn.MultiplicationNode: "*",
    bn.NaturalNode: _val,
    bn.NegateNode: "-",
    bn.NegativeRealNode: _val,
    bn.NormalNode: "Normal",
    bn.HalfNormalNode: "HalfNormal",
    bn.NotEqualNode: "!=",
    bn.NotInNode: "NotIn",
    bn.NotNode: "not",
    bn.Observation: lambda n: f"Observation {str(n.value)}",
    bn.PhiNode: "Phi",
    bn.PositiveRealNode: _val,
    bn.PowerNode: "**",
    bn.ProbabilityNode: _val,
    bn.Query: "Query",
    bn.RealNode: _val,
    bn.RShiftNode: ">>",
    bn.SampleNode: "Sample",
    bn.StudentTNode: "StudentT",
    bn.SwitchNode: "Switch",
    bn.TensorNode: "Tensor",
    bn.ToIntNode: "ToInt",
    bn.ToMatrixNode: "ToMatrix",
    bn.ToNegativeRealNode: "ToNegReal",
    bn.ToPositiveRealMatrixNode: "ToPosRealMatrix",
    bn.ToPositiveRealNode: "ToPosReal",
    bn.ToProbabilityNode: "ToProb",
    bn.ToRealMatrixNode: "ToRealMatrix",
    bn.ToRealNode: "ToReal",
    bn.UniformNode: "Uniform",
    bn.UntypedConstantNode: _val,
    bn.VectorIndexNode: "index",
}

_none = []
_left_right = ["left", "right"]
_operand = ["operand"]
_probability = ["probability"]


def _numbers(n: int) -> List[str]:
    return [str(x) for x in range(n)]


def _numbered_or_left_right(node: bn.BMGNode) -> List[str]:
    if len(node.inputs) == 2:
        return _left_right
    return _numbers(len(node.inputs))


def _prefix_numbered(prefix: List[str]) -> Callable:
    return lambda node: prefix + _numbers(len(node.inputs) - len(prefix))


_edge_labels = {
    bn.AdditionNode: _numbered_or_left_right,
    bn.BernoulliLogitNode: _probability,
    bn.BernoulliNode: _probability,
    bn.BetaNode: ["alpha", "beta"],
    bn.BinomialNode: ["count", "probability"],
    bn.BinomialLogitNode: ["count", "probability"],
    bn.BooleanNode: _none,
    bn.CategoricalNode: _probability,
    bn.Chi2Node: ["df"],
    bn.ChoiceNode: _prefix_numbered(["condition"]),
    bn.ColumnIndexNode: _left_right,
    bn.ComplementNode: _operand,
    bn.ConstantBooleanMatrixNode: _none,
    bn.ConstantNaturalMatrixNode: _none,
    bn.ConstantNegativeRealMatrixNode: _none,
    bn.ConstantPositiveRealMatrixNode: _none,
    bn.ConstantProbabilityMatrixNode: _none,
    bn.ConstantRealMatrixNode: _none,
    bn.ConstantSimplexMatrixNode: _none,
    bn.ConstantTensorNode: _none,
    bn.DirichletNode: ["concentration"],
    bn.DivisionNode: _left_right,
    bn.EqualNode: _left_right,
    bn.ExpM1Node: _operand,
    bn.ExpNode: _operand,
    bn.ExpProductFactorNode: _numbered_or_left_right,
    bn.FlatNode: _none,
    bn.GammaNode: ["concentration", "rate"],
    bn.GreaterThanEqualNode: _left_right,
    bn.GreaterThanNode: _left_right,
    bn.HalfCauchyNode: ["scale"],
    bn.IfThenElseNode: ["condition", "consequence", "alternative"],
    bn.IndexNode: _left_right,
    bn.LessThanEqualNode: _left_right,
    bn.LessThanNode: _left_right,
    bn.Log1mexpNode: _operand,
    bn.LogisticNode: _operand,
    bn.LogNode: _operand,
    bn.LogSumExpNode: _numbered_or_left_right,
    bn.LogSumExpTorchNode: ["operand", "dim", "keepdim"],
    bn.LogSumExpVectorNode: _operand,
    bn.SwitchNode: _numbered_or_left_right,
    bn.MatrixMultiplicationNode: _left_right,
    bn.MatrixScaleNode: _numbered_or_left_right,
    bn.MultiplicationNode: _numbered_or_left_right,
    bn.NaturalNode: _none,
    bn.NegateNode: _operand,
    bn.NegativeRealNode: _none,
    bn.NormalNode: ["mu", "sigma"],
    bn.HalfNormalNode: ["sigma"],
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
    bn.TensorNode: _numbered_or_left_right,
    bn.ToIntNode: _operand,
    bn.ToMatrixNode: _prefix_numbered(["rows", "columns"]),
    bn.ToNegativeRealNode: _operand,
    bn.ToPositiveRealMatrixNode: _operand,
    bn.ToPositiveRealNode: _operand,
    bn.ToProbabilityNode: _operand,
    bn.ToRealMatrixNode: _operand,
    bn.ToRealNode: _operand,
    bn.UniformNode: ["low", "high"],
    bn.VectorIndexNode: _left_right,
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
    return labels(node)[i]
