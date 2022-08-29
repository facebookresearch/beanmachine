# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing
from typing import Callable, List

from torch import Size


def identity_fnc(a: List[int]) -> List[int]:
    return a


def _create_input_list_from_target_list(
    product_list: List[int], input_project_size: List[int]
) -> Callable[[List[int]], int]:
    # given a coordinate index of target, compute a global index of input
    def input_list_from_target_list(target_list: List[int]) -> int:
        i = 0
        j = len(product_list) - 1
        index = 0
        for inx in target_list:
            if input_project_size[i] == 1:
                i = i + 1
                j = j - 1
                continue
            else:
                next = inx * product_list[j]
                index = index + next
            j = j - 1
            i = i + 1
        return index

    return input_list_from_target_list


def _create_target_index_to_composite(
    target_size: Size, group_size: List[int]
) -> Callable[[int], List]:
    # given a global index, produce a coordinate
    def target_index_to_composite(ti: int) -> List:
        index_list = []
        current_index = ti
        j = len(target_size) - 1
        for _ in target_size:
            next_index = math.floor(current_index / group_size[j])
            index_list.append(next_index)
            current_index = current_index % group_size[j]
            j = j - 1
        return index_list

    return target_index_to_composite


def _normalize_size(input_size: Size, target_size: Size) -> List[int]:
    # Make the input size length equal to target size by buffering with 1's
    input_project_size = []
    ones_to_add = len(target_size) - len(input_size)
    for _ in range(0, ones_to_add):
        input_project_size.append(1)
    for dim in input_size:
        input_project_size.append(dim)
    return input_project_size


def broadcast_fnc(input_size: Size, target_size: Size) -> typing.Optional[Callable]:
    if input_size == target_size:
        return identity_fnc

    input_project_size = _normalize_size(input_size, target_size)
    assert len(input_project_size) == len(target_size)

    # the input can be broadcast to the target if
    # input_dim[i] == target_dim[i] || input_dim[i] == 1 for all i
    for i in range(0, len(target_size)):
        if input_project_size[i] != 1 and target_size[i] != input_project_size[i]:
            return None

    # in order to map from a composite index to a coordinate index we
    # need to know how many elements are in each element of each dimension
    # for example, in the case of a list of matrices we might have the size 4 x 3 x 2
    # which means we have a list of 4 elements, where each element is a matrix of 6 elements.
    # Within the matrix, we have 3 elements, each of size 2. In this case, the group size array
    # should be [6, 2, 1]
    group_size = []
    current = 1
    L = len(target_size)
    for k in range(0, L).__reversed__():
        d = target_size[k]
        group_size.append(current)
        current = current * d

    target_index_to_composite = _create_target_index_to_composite(
        target_size, group_size
    )
    # product list should be [2, 1, 1]
    product_list = []
    current = 1
    # the element at index N-j should be the size of the group at dimension j
    # for [1,1,3] we want [1,3,3]. For [3,2,1] we want [1,1,2]
    for k in range(0, len(input_project_size)).__reversed__():
        d = input_project_size[k]
        product_list.append(current)
        current = current * d

    input_list_from_target_list = _create_input_list_from_target_list(
        product_list, input_project_size
    )
    return lambda target_index: input_list_from_target_list(
        target_index_to_composite(target_index)
    )
