# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import namedtuple

from patsy.desc import Evaluator
from patsy.desc import INTERCEPT
from patsy.desc import IntermediateExpr
from patsy.desc import ModelDesc
from patsy.infix_parser import ParseNode
from patsy.infix_parser import infix_parse
from patsy.parse_formula import Operator
from patsy.parse_formula import _atomic_token_types
from patsy.parse_formula import _default_ops
from patsy.parse_formula import _tokenize_formula


RandomEffectsTerm = namedtuple("RandomEffectsTerm", ["expr", "factor"])


def eval_bar(evaluator, tree):
    """Evaluation function for the bar operator AST node."""

    assert len(tree.args) == 2
    expr_node, factor_node = tree.args

    # create model description for the expression left of the bar
    expr_node = ParseNode("~", None, [expr_node], expr_node.origin)
    expr_md = ModelDesc.from_formula(expr_node)

    # create model description for grouping factor right of the bar
    factor_node = ParseNode("~", None, [factor_node], factor_node.origin)
    factor_md = ModelDesc.from_formula(factor_node)
    factor_md.rhs_termlist.remove(INTERCEPT)

    # combine these in a random effects term
    re = RandomEffectsTerm(expr=expr_md, factor=factor_md)

    # return corresponding intermediate expression
    return IntermediateExpr(False, None, False, [re])


def evaluate_formula(formula: str) -> ModelDesc:
    """Given mixed effects formula, return a model description.

    :param formula: mixed effects model formula
    :type formula: str
    :return: model description including outcome variable (lhs) and fixed and/or random effects (rhs)
    :rtype: `class:ModelDesc`
    """

    # mixed effects specific operators
    extra_operators = [Operator("|", 2, 50), Operator("|| ", 2, 50)]

    # construct a list of operator strings needed for tokenization
    operators = _default_ops + extra_operators
    operator_strings = [op.token_type for op in operators]

    tokens = list(_tokenize_formula(formula, operator_strings))
    node = infix_parse(tokens, operators, _atomic_token_types)

    e = Evaluator()

    # we can't handle double bar yet
    e.add_op("|", 2, eval_bar)

    model_desc = e.eval(node, require_evalexpr=False)

    return model_desc
