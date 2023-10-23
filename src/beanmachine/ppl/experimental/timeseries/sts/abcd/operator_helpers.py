# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from copy import deepcopy
from itertools import combinations
from typing import List

from gpytorch.kernels import AdditiveKernel, Kernel, ProductKernel
from sts.abcd.expression import KernelExpression
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.abcd.utils import is_base_kernel
from sts.gp.kernels import ConstantKernel

"""
TODO: This can be reorganized by using parent pointer of KernelExpression.

Each traverse_dfs.. functions are to update the new_kernel_list at each corresponding node.
traverse_bottom_up utilize the new_kernel_list info to get expanded kernels.
e.g:
Suppose we have cur_exp = k0 + k1 * k2, and would like to add kernel_to_add = k4.
The kernel expression is:
     k0 + k1 * k2
    /         \
   k0      k1 * k2
           /    \
           k1   k2

traverse_dfs_add: add kernel_to_add to every node in the kernel expression.
We add k4 to it in new_kernel_list of each node in the tree.
From top down:
              k0 + k1 * k2
            [k0 + k1 * k2 + k4]
            /         \
           k0      k1 * k2
       [k0 + k4]  [k1 * k2 + k4]
                    /    \
                    k1   k2
              [k1 + k4]  [k2 + k4]
traverse_bottom_up: update the new expressions to the root.
From bottom up:
               k0 + k1 * k2
        [k0 + k1 * k2 + k4, k0 + k4 + k1 * k2, k0 + k1 * k2 + k4, k0 + (k1 + k4) * k2, k0 + k1 * (k2 + k4)]
            /         \
           k0      k1 * k2
    [k0 + k4]    [k1 * k2 + k4, (k1 + k4) * k2, k1 * (k2 + k4)]
                    /    \
                    k1   k2
            [k1 + k4]  [k2 + k4]
Therefore, it returns 5 kernels to root, which are our expanded kernels:
[k0 + k1 * k2 + k4, k0 + k4 + k1 * k2, k0 + k1 * k2 + k4, k0 + (k1 + k4) * k2, k0 + k1 * (k2 + k4)].
Other operators perform similarly.
"""


def traverse_dfs_add(cur_exp: KernelExpression, kernel_to_add: Kernel):
    """
    Helper function for S->S+B.
    Update each kernel expression node from top down.
    Update new_kernel_list to kernel + kernel_to_add.

    :param cur_exp: current `KernelExpression` to update.
    :param kernel_to_add: to add base kernel kernel_to_add with kernel.
    """
    cur_exp.new_kernel_list.append(deepcopy(cur_exp.kernel) + deepcopy(kernel_to_add))
    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_add(cur_exp.lhs, kernel_to_add)
        traverse_dfs_add(cur_exp.rhs, kernel_to_add)


def traverse_dfs_mul(cur_exp: KernelExpression, kernel_to_mul: Kernel):
    """
    Helper function for S->S*B.
    Update each kernel expression node from top down.
    Update new_kernel_list to kernel * kernel_to_mul

    :param cur_exp: current `KernelExpression` to update.
    :param kernel_to_mul: to multiply a base kernel kernel_to_mul with kernel.
    """
    cur_exp.new_kernel_list.append(deepcopy(cur_exp.kernel) * deepcopy(kernel_to_mul))
    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_mul(cur_exp.lhs, kernel_to_mul)
        traverse_dfs_mul(cur_exp.rhs, kernel_to_mul)


def traverse_dfs_replace(cur_exp: KernelExpression, base_kernel: Kernel):
    """
    Helper function for B->B1.
    Update each kernel expression node from top down.
    Replace new_kernel_list at leaf which is base kernel with base_kernel

    :param cur_exp: current `KernelExpression` to update.
    :param base_kernel: to replace kernel with base kernel base_kernel.
    """
    if is_base_kernel(cur_exp.kernel):
        cur_exp.new_kernel_list.append(deepcopy(base_kernel))

    else:
        traverse_dfs_replace(cur_exp.lhs, base_kernel)
        traverse_dfs_replace(cur_exp.rhs, base_kernel)


def traverse_dfs_cp(cur_exp: KernelExpression):
    """
    Helper function for S->CP(S,S).
    Update each kernel expression node from top down.
    Add changepoint for new_kernel_list.

    :param cur_exp: current `KernelExpression` to update.
    """
    cur_exp.new_kernel_list.append(
        ChangePointABCDKernel(deepcopy(cur_exp.kernel), deepcopy(cur_exp.kernel))
    )

    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_cp(cur_exp.lhs)
        traverse_dfs_cp(cur_exp.rhs)


def traverse_dfs_cw(cur_exp: KernelExpression, pos_constant_kernel: int):
    """
    Helper function for S->CW(S,S), S->CW(S,C), S->CW(C,S).
    Update each kernel expression node from top down.
    Add changewindow for new_kernel_list.

    :param cur_exp: current `KernelExpression` to update.
    :param pos_constant_kernel: if -1, S->CW(S, C);
                                else if 0, S->CW(S, S);
                                else, S->CW(C, S).
    """
    if pos_constant_kernel == -1:
        cur_exp.new_kernel_list.append(
            ChangeWindowABCDKernel(deepcopy(cur_exp.kernel), ConstantKernel(1.0))
        )

    elif pos_constant_kernel == 0:
        cur_exp.new_kernel_list.append(
            ChangeWindowABCDKernel(deepcopy(cur_exp.kernel), deepcopy(cur_exp.kernel))
        )

    else:
        cur_exp.new_kernel_list.append(
            ChangeWindowABCDKernel(ConstantKernel(1.0), deepcopy(cur_exp.kernel))
        )

    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_cw(cur_exp.lhs, pos_constant_kernel)
        traverse_dfs_cw(cur_exp.rhs, pos_constant_kernel)


def traverse_dfs_mul_const(cur_exp: KernelExpression, kernel_to_mul: Kernel):
    """
    Helper function for S->S*(B+C).
    Update each kernel expression node from top down.
    Update new_kernel_list to kernel * (kernel_to_mul + C).

    :param cur_exp: current `KernelExpression` to update.
    :param kernel_to_mul: a base kernel, multiply (kernel_to_mul + C) with kernel.
    """
    cur_exp.new_kernel_list.append(
        deepcopy(cur_exp.kernel) * (deepcopy(kernel_to_mul) + ConstantKernel(1.0))
    )
    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_mul_const(cur_exp.lhs, kernel_to_mul)
        traverse_dfs_mul_const(cur_exp.rhs, kernel_to_mul)


def traverse_dfs_simplify_base(cur_exp: KernelExpression, to_kernel: Kernel):
    """
    Helper function for S->B.
    Update each kernel expression node from top down.
    Simplify new_kernel_list that are not base kernels to base kernel to_kernel.

    :param cur_exp: current `KernelExpression` to update.
    :param to_kernel: replace kernel to a base kernel to_kernel.
    """
    if not is_base_kernel(cur_exp.kernel):
        cur_exp.new_kernel_list.append(deepcopy(to_kernel))
        traverse_dfs_simplify_base(cur_exp.lhs, to_kernel)
        traverse_dfs_simplify_base(cur_exp.rhs, to_kernel)


def traverse_dfs_simplify_add(cur_exp: KernelExpression):
    """
    Helper function for S+S1->S.
    Update each kernel expression node from top down.
    Simplify new_kernel_list to any summands of it.

    :param cur_exp: current `KernelExpression` to update.
    """
    cur_exp.new_kernel_list = add_combination(cur_exp.kernel)

    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_simplify_add(cur_exp.lhs)
        traverse_dfs_simplify_add(cur_exp.rhs)


def traverse_dfs_simplify_mul(cur_exp: KernelExpression):
    """
    Helper function for S*S1->S.
    Update each kernel expression node from top down.
    Simplify new_kernel_list to any multipliers of it.

    :param cur_exp: current `KernelExpression` to update.
    """
    cur_exp.new_kernel_list = mul_combination(cur_exp.kernel)

    if not is_base_kernel(cur_exp.kernel):
        traverse_dfs_simplify_mul(cur_exp.lhs)
        traverse_dfs_simplify_mul(cur_exp.rhs)


def add_combination(kernel: Kernel):
    """
    Get all combination of summands of an additive kernel.

    :param kernel: the additive kernel.
    :return: all combination of summands.
    """
    res = []
    if isinstance(kernel, AdditiveKernel):
        kernel_summands = kernel.kernels
        if len(kernel_summands) >= 2:
            indexes = list(range(len(kernel_summands)))
            for num_summands in range(1, len(kernel_summands)):
                comb = combinations(indexes, num_summands)
                for index_to_comb in list(comb):
                    kernel = None
                    for j in index_to_comb:
                        if kernel is None:
                            kernel = deepcopy(kernel_summands[j])
                        else:
                            kernel += deepcopy(kernel_summands[j])
                    res.append(kernel)
    return res


def mul_combination(kernel: Kernel):
    """
    Get all combination of multipliers of a product kernel.

    :param kernel: the product kernel.
    :return: all combination of multipliers.
    """
    res = []
    if isinstance(kernel, ProductKernel):
        kernel_multiplier = kernel.kernels
        if len(kernel_multiplier) >= 2:
            indexes = list(range(len(kernel_multiplier)))
            for num_summands in range(1, len(kernel_multiplier)):
                comb = combinations(indexes, num_summands)
                for index_to_comb in list(comb):
                    kernel = None
                    for j in index_to_comb:
                        if kernel is None:
                            kernel = deepcopy(kernel_multiplier[j])
                        else:
                            kernel *= deepcopy(kernel_multiplier[j])
                    res.append(kernel)
    return res


def _compute_res_list(
    cur_exp: KernelExpression,
    cur_exp_child1: KernelExpression,
    cur_exp_child2: KernelExpression,
) -> List[Kernel]:
    """
    Helper function to get all new kernels from cur_exp_child1 side.
    :param cur_exp: `KernelExpression`.
    :param cur_exp_child1: `KernelExpression`, one of two children of cur_exp.
    :param cur_exp_child2: `KernelExpression`, one of two children of cur_exp.
    :return: a list of kernels collected from children for this expression.
    """
    res = []
    child_reslist = traverse_bottom_up(cur_exp_child1)
    for child_res in child_reslist:
        value = deepcopy(child_res)
        if type(cur_exp.kernel) == AdditiveKernel:
            value += deepcopy(cur_exp_child2.kernel)
        elif type(cur_exp.kernel) == ProductKernel:
            value *= deepcopy(cur_exp_child2.kernel)
        elif type(cur_exp.kernel) == ChangePointABCDKernel:
            value = ChangePointABCDKernel(value, deepcopy(cur_exp_child2.kernel))
        elif type(cur_exp.kernel) == ChangeWindowABCDKernel:
            value = ChangeWindowABCDKernel(value, deepcopy(cur_exp_child2.kernel))
        res.append(value)
    return res


def traverse_bottom_up(cur_exp: KernelExpression) -> List[Kernel]:
    """
    Get all possible expanded kernels for each node from bottom up by using new_kernel_list info.
    :param cur_exp: `KernelExpression`.
    :return: a list of kernels collected from children for this expression.
    """

    if cur_exp.lhs is None and cur_exp.rhs is None:
        return cur_exp.new_kernel_list

    res = cur_exp.new_kernel_list
    res.extend(_compute_res_list(cur_exp, cur_exp.lhs, cur_exp.rhs))
    res.extend(_compute_res_list(cur_exp, cur_exp.rhs, cur_exp.lhs))
    return res
