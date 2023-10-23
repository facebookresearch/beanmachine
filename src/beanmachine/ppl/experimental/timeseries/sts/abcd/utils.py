# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List

from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    PeriodicKernel,
    RBFKernel,
    ScaleKernel,
)
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.gp.kernels import ConstantKernel, WhiteNoiseKernel


"""
Base kernels as specified in the paper: WN, LN, SE, PER.
"""
BASE_KERNELS = [
    WhiteNoiseKernel(noise=1e-5),
    LinearKernel(),
    ScaleKernel(RBFKernel()),
    ScaleKernel(PeriodicKernel()),
]
"""
Base kernels and constant kernels as specified in the paper: WN, LN, SE, PER, C.
"""
BASE_CONST_KERNELS = BASE_KERNELS + [ConstantKernel(constant=1.0)]
"""
grammar rules.
"""
BASIC_GRAMMAR_RULES = ["S->S+B", "S->S*B", "B->B1"]
CP_GRAMMAR_RULES = ["S->CP(S,S)", "S->CW(S,S)", "S->CW(S,C)", "S->CW(C,S)"]
HEURISTIC_GRAMMAR_RULES = ["S->S*(B+C)", "S->B", "S+S1->S", "S*S1->S"]
GRAMMAR_RULES = {
    "basic": BASIC_GRAMMAR_RULES,
    "cp": CP_GRAMMAR_RULES,
    "heuristic": HEURISTIC_GRAMMAR_RULES,
    "all": BASIC_GRAMMAR_RULES + CP_GRAMMAR_RULES + HEURISTIC_GRAMMAR_RULES,
}


def is_base_kernel(kernel: Kernel) -> bool:
    """
    If kernel is a base kernel, return True; else, return False.

    :param kernel: kernel to check.
    :return: True if kernel is a base kernel; False otherwise.
    """
    return type(kernel) in map(type, BASE_CONST_KERNELS)


def remove_redundancy(kernel_list: List[Kernel]) -> List[Kernel]:
    """
    Remove duplicate kernels in kernel_list.

    :param kernel_list: the kernel list to remove redundancy.
    :return: the list of kernels without duplication.
    """
    kernel_set = set(kernel_list)
    for i in range(len(kernel_list) - 1):
        for j in range(i + 1, len(kernel_list)):
            kernel_map = {}
            if is_kernel_type_eq(kernel_list[i], kernel_list[j], kernel_map):
                if kernel_list[j] in kernel_set:
                    kernel_set.remove(kernel_list[j])
                break
    return list(kernel_set)


def _is_base_kernel_type_eq(kernel1: Kernel, kernel2: Kernel) -> bool:
    """
    Check whether two base kernels are of the same type.

    :param kernel1: one base kernel.
    :param kernel2: another base kernel.
    :return: True if kernel1 and kernel2 are identical, False otherwise.
    """
    if type(kernel1) == type(kernel2) and type(kernel1) != ScaleKernel:
        return True
    elif type(kernel1) == type(kernel2) and type(kernel1.base_kernel) == type(
        kernel2.base_kernel
    ):
        return True
    else:
        return False


def _init_kernel_map(kernel_map: Dict):
    """
    Initiate kernel_map of None to {}.
    """
    if kernel_map is None:
        return {}
    return kernel_map


def is_kernel_type_eq(
    kernel1: Kernel, kernel2: Kernel, kernel_map: Dict = None
) -> bool:
    """
    Check whether two kernels are of the same type. Two kernels are of the same type if they
    have the same type and every subkernel is of the same type.

    :param kernel1: one kernel.
    :param kernel2: another kernel.
    :param kernel_map: the dictionary of mapping every subkernel in kernel1 to kernel2.
    :return: True if kernel1 and kernel2 are identical, False otherwise.
    """
    if is_base_kernel(kernel1) and is_base_kernel(kernel2):
        return _is_base_kernel_type_eq(kernel1, kernel2)

    elif is_base_kernel(kernel1) or is_base_kernel(kernel2):
        return False

    if type(kernel1) != type(kernel2) or len(kernel1.kernels) != len(kernel2.kernels):
        return False

    kernel_map = _init_kernel_map(kernel_map)

    # ChangeKernel should care the order of children
    # CW(S, C) and CW(C, S) are different
    if isinstance(kernel1, (ChangePointABCDKernel, ChangeWindowABCDKernel)):
        if is_kernel_type_eq(
            kernel1.kernels[0], kernel2.kernels[0], kernel_map
        ) and is_kernel_type_eq(kernel1.kernels[1], kernel2.kernels[1], kernel_map):
            kernel_map[kernel1] = kernel2
            return True
        else:
            return False
    else:
        children1 = set(kernel1.kernels)
        children2 = set(kernel2.kernels)

        for c1 in children1:
            if c1 in kernel_map.keys():
                if kernel_map[c1] in children2:
                    children2.remove(kernel_map[c1])
                    continue
                else:
                    return False
            map_node2 = None
            for c2 in children2:
                if is_kernel_type_eq(c1, c2, kernel_map):
                    map_node2 = c2
                    break
            if map_node2 is None:
                return False
            children2.remove(map_node2)
        kernel_map[kernel1] = kernel2
        return True
