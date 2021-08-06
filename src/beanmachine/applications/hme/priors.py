# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import beanmachine.graph as bmgraph


DIST_TYPE_DICT = {
    # prob
    "beta": bmgraph.DistributionType.BETA,
    # pos_real
    "gamma": bmgraph.DistributionType.GAMMA,
    "half_cauchy": bmgraph.DistributionType.HALF_CAUCHY,
    "half_normal": bmgraph.DistributionType.HALF_NORMAL,
    # real
    "normal": bmgraph.DistributionType.NORMAL,
    "t": bmgraph.DistributionType.STUDENT_T,
    # prob/pos_real/real
    "flat": bmgraph.DistributionType.FLAT,
    # natural
    "binomial": bmgraph.DistributionType.BINOMIAL,
    "categorical": bmgraph.DistributionType.CATEGORICAL,
}

SAMPLE_TYPE_DICT = {
    # prob
    "beta": "prob",
    # pos_real
    "gamma": "pos_real",
    "half_cauchy": "pos_real",
    "half_normal": "pos_real",
    # real
    "normal": "real",
    "t": "real",
    # prob/pos_real/real
    "flat": "real",  # FIXME: more flexible support of flat prior
    # natural
    "binomial": "natural",
    "categorical": "natural",
}

PARAM_TYPE_DICT = {
    "beta": {"alpha": "pos_real", "beta": "pos_real"},
    "gamma": {"alpha": "pos_real", "beta": "pos_real"},
    "half_cauchy": {"scale": "pos_real"},
    "half_normal": {"scale": "pos_real"},
    "normal": {"mean": "real", "scale": "pos_real"},
    "t": {"dof": "pos_real", "mean": "real", "scale": "pos_real"},
    "flat": {},
    "binomial": {"total_count": "natural", "prob": "prob"},
    "categorical": {"p_vec": "simplex"},
}

ATOMIC_TYPE_DICT = {
    "real": bmgraph.AtomicType.REAL,
    "pos_real": bmgraph.AtomicType.POS_REAL,
    "prob": bmgraph.AtomicType.PROBABILITY,
    "natural": bmgraph.AtomicType.NATURAL,
}
