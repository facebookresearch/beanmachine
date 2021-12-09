# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def dict_from_first_to_second_component(tuples):
    return {t[0]: t[1] for t in tuples}
