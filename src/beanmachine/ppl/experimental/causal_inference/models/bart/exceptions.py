# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class TreeStructureError(Exception):
    """Base class for errors related to tree structure"""

    pass


class PruneError(TreeStructureError):
    """Raised for errors in pruning operations on a tree such as trying to prune
    a root node or trying to prune a node which would has non-terminal children."""

    pass


class GrowError(TreeStructureError):
    """Raised for errors in growing a tree such as trying to grow from a node along
    an input dimension which has no unique values."""

    pass


class NotInitializedError(AttributeError):
    """Raised for errors in accessing model attributes which have not been initialized
    for example trying to predict a model which has not been trained."""

    pass
