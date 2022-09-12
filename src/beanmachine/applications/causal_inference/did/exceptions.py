# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class DataException(Exception):
    pass


class InterventionException(DataException):
    pass


class PrePeriodException(DataException):
    pass


class PostPeriodException(DataException):
    pass


class DataColumnException(DataException):
    pass


class ModelException(Exception):
    pass


class ModelNotFitException(ModelException):
    pass


class PriorConfigException(Exception):
    pass
