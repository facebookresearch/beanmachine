# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import itertools
from copy import deepcopy
from typing import List

from gpytorch.kernels import (
    AdditiveKernel,
    Kernel,
    LinearKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.abcd.utils import is_base_kernel
from sts.gp.kernels import ConstantKernel, WhiteNoiseKernel

"""
TODO: This can be reorganzied using parent node instead of new_kernel_list.
"""


class KernelExpression:
    """
    Class used to store kernel information and structure as a binary tree node.

    :param kernel: class `Kernel`. This is the kernel that this `KernelExpression` would help store info.
    :param parent: class `KernelExpression`. This is the parent of this node.
    :param new_kernel_list: class `List[Kernel]`. This is the list to store expansion of kernel.
    :param children: class `List[KernelExpression]`. This is the list of children of this node. List size is 2.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.new_kernel_list = []
        self.lhs = None
        self.rhs = None
        self.init_children()

    def init_children(self):
        """
        Initialize the list of children.
        :return: The list of children, if it's an internal node (composite kernel);
                 None, if it's a leaf node (base kernel).
        """
        if not is_base_kernel(self.kernel):
            num_children = len(self.kernel.kernels)
            if num_children == 2:
                self.lhs = KernelExpression(self.kernel.kernels[0])
                self.rhs = KernelExpression(self.kernel.kernels[1])
            else:
                left_kernel = self.kernel.kernels[0]
                if type(self.kernel) == AdditiveKernel:
                    for i in range(1, num_children - 1):
                        left_kernel += self.kernel.kernels[i]
                    self.lhs = KernelExpression(left_kernel)
                    self.rhs = KernelExpression(self.kernel.kernels[num_children - 1])

                else:
                    for i in range(1, num_children - 1):
                        left_kernel *= self.kernel.kernels[i]
                    self.lhs = KernelExpression(left_kernel)
                    self.rhs = KernelExpression(self.kernel.kernels[num_children - 1])

    def __repr__(self):
        if is_base_kernel(self.kernel):
            return _get_kernel_name(self.kernel)
        elif isinstance(self.kernel, AdditiveKernel):
            return f"({repr(self.lhs)} + {repr(self.rhs)})"
        elif isinstance(self.kernel, ProductKernel):
            return f"({repr(self.lhs)} x {repr(self.rhs)})"
        elif isinstance(self.kernel, ChangePointABCDKernel):
            return f"CP({repr(self.lhs)}, {repr(self.rhs)})"
        elif isinstance(self.kernel, ChangeWindowABCDKernel):
            return f"CW({repr(self.lhs)}, {repr(self.rhs)})"

    def additive_form_kernel(self) -> Kernel:
        """
        Get the kernel after apply distributive law to self.kernel.
        :return: the kernel after apply distributive law.
        """
        return _distribute_products(self.kernel)


def _get_kernel_name(kernel):
    KERNEL_NAME = {
        WhiteNoiseKernel: "WN",
        ConstantKernel: "C",
        RBFKernel: "RBF",
        PeriodicKernel: "PER",
        LinearKernel: "LIN",
    }
    if type(kernel) in KERNEL_NAME.keys():
        return KERNEL_NAME[type(kernel)]
    elif isinstance(kernel, ScaleKernel):
        return _get_kernel_name(kernel.base_kernel)


def _break_into_summands(kernel) -> List[Kernel]:
    """
    Get the summands of a kernel.
    :param kernel: kernel to get summands.
    :return: the list of summand kernels.
    """
    k = deepcopy(kernel)
    k_dist = _distribute_products(k)

    if isinstance(k_dist, AdditiveKernel):
        return k_dist.kernels
    else:
        return [k_dist]


def _distribute_products(kernel: Kernel) -> Kernel:
    """
    Apply the distributive law to a composite kernel.
    :param kernel: this is kernel to apply distributive law.
    :return: the kernel which is the same as the input kernel, but in additive form.
    """
    k = deepcopy(kernel)
    if isinstance(k, ProductKernel):
        distributed_ops = [
            _break_into_summands(child_kernel) for child_kernel in k.kernels
        ]
        res_k_sum = []

        for prod in itertools.product(*distributed_ops):
            res_k_sum.append(_mul_kernels(prod))
        return _add_kernels(res_k_sum)

    elif isinstance(k, AdditiveKernel):
        return _add_kernels(
            [
                subchild
                for child in k.kernels
                for subchild in _break_into_summands(child)
            ]
        )
    elif isinstance(k, (ChangePointABCDKernel, ChangeWindowABCDKernel)):
        if len(k.kernels) == 2:
            summands = []
            # the constant kernel below is zerokernel, which has no effect on the computation.
            operands_list = [
                [op, ConstantKernel(constant=0.0)]
                for op in _break_into_summands(k.kernels[0])
            ]
            for ops in operands_list:
                k_new = deepcopy(k)
                k_new.kernels = ops
                summands.append(k_new)
            operands_list = [
                [ConstantKernel(constant=0.0), op]
                for op in _break_into_summands(k.kernels[1])
            ]
            for ops in operands_list:
                k_new = deepcopy(k)
                k_new.kernels = ops
                summands.append(k_new)
            return _add_kernels(summands)
        else:
            raise RuntimeError("Wrong form")
    else:
        return k


def _mul_kernels(kernels: List[Kernel]) -> Kernel:
    """
    Multiply all kernels in the list.
    :return: the kernel by multiplying all kernels.
    """
    res_k = None
    for k in kernels:
        if res_k is None:
            res_k = k
        else:
            res_k *= k
    return res_k


def _add_kernels(kernels: List[Kernel]) -> Kernel:
    """
    Add all kernels in the list.
    :return: the kernel by adding all kernels.
    """
    res_k = None
    for k in kernels:
        if res_k is None:
            res_k = k
        else:
            res_k += k
    return res_k
