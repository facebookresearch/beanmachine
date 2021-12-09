# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.model.statistical_model import random_variable, functional

allowed_functions = {dict, list, set, super, random_variable, functional}

# TODO: Allowing these constructions raises additional problems that
# we have not yet solved. For example, what happens if someone
# searches a list for a value, but the list contains a graph node?
# And so on.
