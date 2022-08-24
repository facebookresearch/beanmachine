# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import collections
import typing
from typing import Callable, Dict, List, Optional, Type

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.execution_context import ExecutionContext, FunctionCall
from beanmachine.ppl.compiler.sizer import Sizer

TransformAssessment = collections.namedtuple(
    "TransformAssessment", ["node_needs_transform", "error_report"]
)


def flatten(
    inputs: List[Optional[typing.Union[bn.BMGNode, List[bn.BMGNode]]]]
) -> List[bn.BMGNode]:
    parents = []
    for input in inputs:
        if input is None:
            continue
        if isinstance(input, List):
            for i in input:
                parents.append(i)
        else:
            parents.append(input)
    return parents


class Cloner:
    def __init__(self, original: BMGraphBuilder):
        self.bmg_original = original
        self.bmg = BMGraphBuilder(ExecutionContext())
        self.bmg._fix_observe_true = self.bmg_original._fix_observe_true
        self.sizer = Sizer()
        self.node_factories = _node_factories(self.bmg)
        self.value_factories = _constant_factories(self.bmg)
        self.copy_context = {}

    def clone(self, original: bn.BMGNode, parents: List[bn.BMGNode]) -> bn.BMGNode:
        if self.value_factories.__contains__(type(original)):
            if isinstance(original, bn.ConstantNode):
                image = self.value_factories[type(original)](original.value)
            else:
                raise ValueError(
                    f"Internal compiler error. The type f{type(original)} should not be in the value factory because it does not have a value attribute"
                )
        elif isinstance(original, bn.Observation):
            assert len(parents) == 1
            sample = parents[0]
            if isinstance(sample, bn.SampleNode):
                image = self.bmg.add_observation(sample, original.value)
            else:
                raise ValueError("observations must have a sample operand")
        elif isinstance(original, bn.Query):
            assert len(parents) == 1
            return self.bmg.add_query(parents[0], original.rv_identifier)
        elif isinstance(original, bn.TensorNode):
            image = self.bmg.add_tensor(self.sizer[original], *parents)
        else:
            image = self.node_factories[type(original)](*parents)

        locations = self.bmg_original.execution_context.node_locations(original)
        for site in locations:
            new_args = []
            for arg in site.args:
                if self.copy_context.__contains__(arg):
                    new_args.append(self.copy_context[arg])
                else:
                    # TODO: error out instead? it's possible that multiple nodes replace a single node
                    new_args.append(arg)
            new_site = FunctionCall(site.func, new_args, {})
            self.bmg.execution_context.record_node_call(image, new_site)
        self.copy_context[original] = image
        return image


def _node_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    return {
        bn.BernoulliLogitNode: bmg.add_bernoulli_logit,
        bn.BernoulliNode: bmg.add_bernoulli,
        bn.BetaNode: bmg.add_beta,
        bn.BinomialNode: bmg.add_binomial,
        bn.BinomialLogitNode: bmg.add_binomial_logit,
        bn.CategoricalNode: bmg.add_categorical,
        bn.CategoricalLogitNode: bmg.add_categorical_logit,
        bn.Chi2Node: bmg.add_chi2,
        bn.DirichletNode: bmg.add_dirichlet,
        bn.GammaNode: bmg.add_gamma,
        bn.HalfCauchyNode: bmg.add_halfcauchy,
        bn.HalfNormalNode: bmg.add_halfnormal,
        bn.NormalNode: bmg.add_normal,
        bn.PoissonNode: bmg.add_poisson,
        bn.StudentTNode: bmg.add_studentt,
        bn.UniformNode: bmg.add_uniform,
        bn.AdditionNode: bmg.add_addition,
        bn.BitAndNode: bmg.add_bitand,
        bn.BitOrNode: bmg.add_bitor,
        bn.BitXorNode: bmg.add_bitxor,
        bn.CholeskyNode: bmg.add_cholesky,
        bn.ColumnIndexNode: bmg.add_column_index,
        bn.ComplementNode: bmg.add_complement,
        bn.DivisionNode: bmg.add_division,
        bn.ElementwiseMultiplyNode: bmg.add_elementwise_multiplication,
        bn.EqualNode: bmg.add_equal,
        bn.Exp2Node: bmg.add_exp2,
        bn.ExpNode: bmg.add_exp,
        bn.ExpM1Node: bmg.add_expm1,
        bn.ExpProductFactorNode: bmg.add_exp_product,
        bn.GreaterThanNode: bmg.add_greater_than,
        bn.GreaterThanEqualNode: bmg.add_greater_than_equal,
        bn.IfThenElseNode: bmg.add_if_then_else,
        bn.IsNode: bmg.add_is,
        bn.IsNotNode: bmg.add_is_not,
        bn.ItemNode: bmg.add_item,
        bn.IndexNode: bmg.add_index,
        bn.InNode: bmg.add_in,
        bn.InvertNode: bmg.add_invert,
        bn.LessThanNode: bmg.add_less_than,
        bn.LessThanEqualNode: bmg.add_less_than_equal,
        bn.LogAddExpNode: bmg.add_logaddexp,
        bn.LogisticNode: bmg.add_logistic,
        bn.Log10Node: bmg.add_log10,
        bn.Log1pNode: bmg.add_log1p,
        bn.Log2Node: bmg.add_log2,
        bn.Log1mexpNode: bmg.add_log1mexp,
        bn.LogSumExpVectorNode: bmg.add_logsumexp_vector,
        bn.LogProbNode: bmg.add_log_prob,
        bn.LogNode: bmg.add_log,
        bn.LogSumExpTorchNode: bmg.add_logsumexp_torch,
        bn.LShiftNode: bmg.add_lshift,
        bn.MatrixAddNode: bmg.add_matrix_addition,
        bn.MatrixExpNode: bmg.add_matrix_exp,
        bn.MatrixMultiplicationNode: bmg.add_matrix_multiplication,
        bn.MatrixScaleNode: bmg.add_matrix_scale,
        bn.MatrixSumNode: bmg.add_matrix_sum,
        bn.ModNode: bmg.add_mod,
        bn.MultiplicationNode: bmg.add_multiplication,
        bn.NegateNode: bmg.add_negate,
        bn.NotEqualNode: bmg.add_not_equal,
        bn.NotNode: bmg.add_not,
        bn.NotInNode: bmg.add_not_in,
        bn.PhiNode: bmg.add_phi,
        bn.PowerNode: bmg.add_power,
        bn.RShiftNode: bmg.add_rshift,
        bn.SampleNode: bmg.add_sample,
        bn.SquareRootNode: bmg.add_squareroot,
        bn.SwitchNode: bmg.add_switch,
        bn.SumNode: bmg.add_sum,
        bn.ToMatrixNode: bmg.add_to_matrix,
        bn.ToPositiveRealMatrixNode: bmg.add_to_positive_real_matrix,
        bn.ToRealMatrixNode: bmg.add_to_real_matrix,
        bn.TransposeNode: bmg.add_transpose,
        bn.ToPositiveRealNode: bmg.add_to_positive_real,
        bn.ToRealNode: bmg.add_to_real,
        bn.VectorIndexNode: bmg.add_vector_index,
    }


def _constant_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    return {
        bn.NegativeRealNode: bmg.add_neg_real,
        bn.NaturalNode: bmg.add_natural,
        bn.ConstantNode: bmg.add_constant,
        bn.RealNode: bmg.add_real,
        bn.PositiveRealNode: bmg.add_pos_real,
        bn.ProbabilityNode: bmg.add_probability,
        bn.ConstantBooleanMatrixNode: bmg.add_boolean_matrix,
        bn.ConstantNaturalMatrixNode: bmg.add_natural_matrix,
        bn.ConstantNegativeRealMatrixNode: bmg.add_neg_real_matrix,
        bn.ConstantSimplexMatrixNode: bmg.add_simplex,
        bn.ConstantPositiveRealMatrixNode: bmg.add_pos_real_matrix,
        bn.ConstantRealMatrixNode: bmg.add_real_matrix,
        bn.ConstantTensorNode: bmg.add_constant_tensor,
        bn.UntypedConstantNode: bmg.add_constant,
    }


class NodeTransformer:
    def assess_node(
        self, node: bn.BMGNode, original: BMGraphBuilder
    ) -> TransformAssessment:
        raise NotImplementedError("this is an abstract base class")

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(
        self, node: bn.BMGNode, new_inputs: List[bn.BMGNode]
    ) -> typing.Optional[typing.Union[bn.BMGNode, List[bn.BMGNode]]]:
        raise NotImplementedError("this is an abstract base class")


def copy_and_replace(
    bmg_original: BMGraphBuilder,
    transformer_creator: Callable[[Cloner, Sizer], NodeTransformer],
) -> typing.Tuple[BMGraphBuilder, ErrorReport]:
    cloner = Cloner(bmg_original)
    transformer = transformer_creator(cloner, cloner.sizer)
    for original in bmg_original.all_nodes():
        inputs = []
        for c in original.inputs.inputs:
            inputs.append(cloner.copy_context[c])
        assessment = transformer.assess_node(original, cloner.bmg_original)

        if len(assessment.error_report.errors) > 0:
            return cloner.bmg, assessment.error_report
        elif assessment.node_needs_transform:
            image = transformer.transform_node(original, inputs)
        else:
            parents = flatten(inputs)
            image = cloner.clone(original, parents)
        if not cloner.copy_context.__contains__(original):
            cloner.copy_context[original] = image
    return cloner.bmg, ErrorReport()
