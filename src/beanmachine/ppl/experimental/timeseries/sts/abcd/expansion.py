# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from copy import deepcopy
from typing import List, Tuple

from gpytorch.kernels import (
    AdditiveKernel,
    Kernel,
    PeriodicKernel,
    ProductKernel,
    ScaleKernel,
)
from sts.abcd.expression import KernelExpression
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.abcd.operator_helpers import traverse_bottom_up
from sts.abcd.operators import (
    B_to_B1,
    S_add_S1_to_S,
    S_mul_S1_to_S,
    S_to_B,
    S_to_CP_S_S,
    S_to_CW_S,
    S_to_S_add_B,
    S_to_S_mul_B,
    S_to_S_mul_B_add_C,
)
from sts.abcd.utils import GRAMMAR_RULES, is_base_kernel, remove_redundancy
from sts.gp.kernels import ConstantKernel, WhiteNoiseKernel


def expand_kernel(
    kernel: Kernel, grammar: List[str] = GRAMMAR_RULES["all"]
) -> List[Kernel]:
    """
    Propose a list of kernels to prepare for search of next level of depth by operating all operators.

    :param kernel: The current kernel to expand.
    :return: a list of kernels that will be optimized and evaluated.
    """
    res = []
    # all operatos

    if "S->S+B" in grammar:
        # S->S+B
        res.extend(S_to_S_add_B(kernel))

    if "S->S*B" in grammar:
        # S->S*B
        res.extend(S_to_S_mul_B(kernel))

    if "B->B1" in grammar:
        # B->B' B is any base kernel
        # kernel is base kernel, is is redundant with S->B
        res.extend(B_to_B1(kernel))

    if "S->CP(S,S)" in grammar:
        # change points
        # S->CP(S,S)
        res.extend(S_to_CP_S_S(kernel))

    if "S->CW(S,S)" in grammar:
        # S->CW(S,S)
        res.extend(S_to_CW_S(kernel, 0))

    if "S->CW(S,C)" in grammar:
        # S->CW(S, C)
        res.extend(S_to_CW_S(kernel, -1))

    if "S->CW(C,S)" in grammar:
        # S->CW(C, S)
        res.extend(S_to_CW_S(kernel, 1))

    if "S->S*(B+C)" in grammar:
        # S->S*(B+C)
        res.extend(S_to_S_mul_B_add_C(kernel))

    if "S->B" in grammar:
        # S->B
        res.extend(S_to_B(kernel))

    if "S+S1->S" in grammar:
        # S+S1->S
        res.extend(S_add_S1_to_S(kernel))
    if "S*S1->S" in grammar:
        # S*S1->S
        res.extend(S_mul_S1_to_S(kernel))

    res = remove_redundancy(res)
    return res


def _get_res_kernel(
    res_kernel: Kernel, child_simp: Kernel, op: str, exclude_wn_const: bool = False
) -> Tuple[Kernel, int, int]:
    """
    Operate child_simp on res_kernel. Helper function for simplify.
    :param res_kernel: the kernel to add or multiply.
    :param child_simp: the kernel to add or multiply.
    :param op: operation add '+' or multiply '*'.
    :param exclude_wn_const: if True, not add/multiply white noise kernel/constant kernel to res_kernel.
    :return: the result kernel, the count of white noise kernel, and the count of constant kernel.
    """
    wn = 0
    cons = 0
    if isinstance(child_simp, WhiteNoiseKernel):
        wn = 1
    elif isinstance(child_simp, ConstantKernel):
        cons = 1

    if exclude_wn_const and (wn == 1 or cons == 1):
        return res_kernel, wn, cons

    if res_kernel is None:
        res_kernel = child_simp
    else:
        if op == "+":
            res_kernel += child_simp
        elif op == "*":
            res_kernel *= child_simp

    return res_kernel, wn, cons


def simplify(kernel: Kernel) -> Kernel:
    """
    Simplify unnecessary structure of a composite kernel.
        WN + WN -> WN
        WN * WN -> WN
        Const + Const -> Const
        Const * S -> S
    :param kernel: The current kernel to expand.
    :return: The simplified kernel.
    """

    if is_base_kernel(kernel):
        return kernel

    wn_cnt = 0
    const_cnt = 0
    res_kernel = None
    if isinstance(kernel, AdditiveKernel):
        for child_kernel in kernel.kernels:
            child_simp = simplify(child_kernel)
            res_kernel, wn, cons = _get_res_kernel(res_kernel, child_simp, "+", True)
            wn_cnt += wn
            const_cnt += cons

        if wn_cnt > 0:
            res_kernel, wn, cons = _get_res_kernel(
                res_kernel, WhiteNoiseKernel(noise=1e-4), "+"
            )
        if const_cnt > 0:
            res_kernel, wn, cons = _get_res_kernel(
                res_kernel, ConstantKernel(constant=1.0), "+"
            )
        return res_kernel
    elif isinstance(kernel, ProductKernel):
        for child_kernel in kernel.kernels:
            child_simp = simplify(child_kernel)
            res_kernel, wn, cons = _get_res_kernel(res_kernel, child_simp, "*", True)
            wn_cnt += wn
            const_cnt += cons

        if wn_cnt > 0:
            res_kernel, wn, cons = _get_res_kernel(
                res_kernel, WhiteNoiseKernel(noise=1e-4), "*"
            )
        if const_cnt > 0 and res_kernel is None:
            res_kernel = ConstantKernel(constant=1.0)
        return res_kernel
    elif isinstance(kernel, ChangeWindowABCDKernel):
        child_simp0 = simplify(kernel.kernels[0])
        child_simp1 = simplify(kernel.kernels[1])
        return ChangeWindowABCDKernel(
            deepcopy(child_simp0),
            deepcopy(child_simp1),
            location=kernel.location.detach().squeeze(-1),
            steep=kernel.steep.detach().squeeze(-1),
        )
    elif isinstance(kernel, ChangePointABCDKernel):
        child_simp0 = simplify(kernel.kernels[0])
        child_simp1 = simplify(kernel.kernels[1])
        return ChangePointABCDKernel(
            deepcopy(child_simp0),
            deepcopy(child_simp1),
            location=kernel.location.item(),
            steep=kernel.steep.item(),
        )


def traverse_kernel_initialize(cur_exp: KernelExpression, value_list: Tuple[float]):
    """
    Update the kernel expression for each periodic kernel in a given kernel expression.
    :param kernel: the kernel expression.
    :parma value_list: the candidate list of period_length.
    """
    if is_base_kernel(cur_exp.kernel):
        cur_exp.new_kernel_list.extend(
            initialize_params_period(cur_exp.kernel, value_list)
        )
    else:
        traverse_kernel_initialize(cur_exp.lhs, value_list)
        traverse_kernel_initialize(cur_exp.rhs, value_list)


def initialize_params_period(kernel: Kernel, value_list: Tuple[float]) -> List[Kernel]:
    """
    Initialize the period_length for a periodic kernel.
    :param kernel: the periodic kernel.
    :parma value_list: the candidate list of period_length.
    :return: list of kernels with initialized period lengths.
    """
    kernel_list = []
    if isinstance(kernel, ScaleKernel):
        if isinstance(kernel.base_kernel, PeriodicKernel):
            for value in value_list:
                kernel.base_kernel.period_length = value
                kernel_list.append(deepcopy(kernel))

    return kernel_list


def initialize_kernel(kernel: Kernel, value_list: Tuple[float]) -> List[Kernel]:
    """
    Initialize a kernel by initialize each periodic kernel in it with period_length.
    :param kernel: the periodic kernel.
    :parma value_list: the candidate list of period_length.
    :return: list of kernels with periodic kernels initialized period lengths.
    """
    cur_exp = KernelExpression(kernel)
    traverse_kernel_initialize(cur_exp, value_list)
    return traverse_bottom_up(cur_exp)
