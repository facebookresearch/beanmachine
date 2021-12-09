# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import beanmachine.graph as bmgraph


class ParamType(Enum):
    REAL = ("real", bmgraph.AtomicType.REAL)
    POS_REAL = ("pos_real", bmgraph.AtomicType.POS_REAL)
    PROB = ("prob", bmgraph.AtomicType.PROBABILITY)
    NATURAL = ("natural", bmgraph.AtomicType.NATURAL)
    COL_SIMPLEX_MATRIX = ("simplex", bmgraph.AtomicType.NATURAL)

    def __init__(self, str_name, atomic_type):
        self.str_name = str_name
        self.atomic_type = atomic_type

    @classmethod
    def match_str(cls, label):
        for param_type in cls:
            if param_type.str_name == label:
                return param_type
        else:
            raise ValueError("Unknown parameter type: '{s}'!".format(s=label))


class Distribution(Enum):
    BETA = (
        "beta",
        bmgraph.DistributionType.BETA,
        ParamType.PROB,
        ["alpha", "beta"],
        {"alpha": ParamType.POS_REAL, "beta": ParamType.POS_REAL},
    )
    GAMMA = (
        "gamma",
        bmgraph.DistributionType.GAMMA,
        ParamType.POS_REAL,
        ["alpha", "beta"],
        {"alpha": ParamType.POS_REAL, "beta": ParamType.POS_REAL},
    )
    HALF_CAUCHY = (
        "half_cauchy",
        bmgraph.DistributionType.HALF_CAUCHY,
        ParamType.POS_REAL,
        ["scale"],
        {"scale": ParamType.POS_REAL},
    )
    HALF_NORMAL = (
        "half_normal",
        bmgraph.DistributionType.HALF_NORMAL,
        ParamType.POS_REAL,
        ["scale"],
        {"scale": ParamType.POS_REAL},
    )
    NORMAL = (
        "normal",
        bmgraph.DistributionType.NORMAL,
        ParamType.REAL,
        ["mean", "scale"],
        {"mean": ParamType.REAL, "scale": ParamType.POS_REAL},
    )
    STUDENT_T = (
        "t",
        bmgraph.DistributionType.STUDENT_T,
        ParamType.REAL,
        ["dof", "mean", "scale"],
        {
            "dof": ParamType.POS_REAL,
            "mean": ParamType.REAL,
            "scale": ParamType.POS_REAL,
        },
    )
    FLAT = (
        "flat",
        bmgraph.DistributionType.FLAT,
        ParamType.REAL,
        [],
        {},
    )
    BINOMIAL = (
        "binomial",
        bmgraph.DistributionType.BINOMIAL,
        ParamType.NATURAL,
        ["total_count", "prob"],
        {"total_count": ParamType.NATURAL, "prob": ParamType.PROB},
    )
    CATEGORICAL = (
        "categorical",
        bmgraph.DistributionType.CATEGORICAL,
        ParamType.NATURAL,
        ["p_vec"],
        {"p_vec": ParamType.COL_SIMPLEX_MATRIX},
    )

    def __init__(self, str_name, dist_type, sample_type, param_order, params):
        self.str_name = str_name
        self.dist_type = dist_type
        self.sample_type = sample_type
        self.param_order = param_order
        self.params = params

    @classmethod
    def match_str(cls, label):
        for dist in cls:
            if dist.str_name == label:
                return dist
        else:
            raise ValueError("Unknown distribution type: '{s}'!".format(s=label))
