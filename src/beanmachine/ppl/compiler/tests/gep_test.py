# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference import BMGInference

trials = torch.tensor([29854.0, 2016.0])
pos = torch.tensor([4.0, 0.0])
buck_rep = torch.tensor([0.0006, 0.01])
n_buckets = len(trials)


def log1mexp(x):
    return (1 - x.exp()).log()


@bm.random_variable
def eta():  # k reals
    return dist.Normal(0.0, 1.0).expand((n_buckets,))


@bm.random_variable
def alpha():  # atomic R+
    return dist.half_normal.HalfNormal(5.0)


@bm.random_variable
def sigma():  # atomic R+
    return dist.half_normal.HalfNormal(1.0)


@bm.random_variable
def length_scale():  # R+
    return dist.half_normal.HalfNormal(0.1)


@bm.functional
def cholesky():  # k by k reals
    delta = 1e-3
    alpha_sq = alpha() * alpha()
    rho_sq = length_scale() * length_scale()
    cov = (buck_rep - buck_rep.unsqueeze(-1)) ** 2
    cov = alpha_sq * torch.exp(-cov / (2 * rho_sq))
    cov += torch.eye(buck_rep.size(0)) * delta
    return torch.linalg.cholesky(cov)


@bm.random_variable
def prev():  # k reals
    return dist.Normal(torch.matmul(cholesky(), eta()), sigma())


@bm.random_variable
def bucket_prob():  # atomic bool
    phi_prev = dist.Normal(0, 1).cdf(prev())  # k probs
    log_prob = pos * torch.log(phi_prev)
    log_prob += (trials - pos) * torch.log1p(-phi_prev)
    joint_log_prob = log_prob.sum()
    # Convert the joint log prob to a log-odds.
    logit_prob = joint_log_prob - log1mexp(joint_log_prob)
    return dist.Bernoulli(logits=logit_prob)


class GEPTest(unittest.TestCase):
    def test_gep_model_compilation(self) -> None:
        self.maxDiff = None
        queries = [prev()]
        observations = {bucket_prob(): torch.tensor([1.0])}
        observed = BMGInference().to_dot(queries, observations)

        expected = """
digraph "graph" {
  N00[label=5.0];
  N01[label=HalfNormal];
  N02[label=Sample];
  N03[label=0.10000000149011612];
  N04[label=HalfNormal];
  N05[label=Sample];
  N06[label=0.0];
  N07[label=1.0];
  N08[label=Normal];
  N09[label=Sample];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=HalfNormal];
  N13[label=Sample];
  N14[label="*"];
  N15[label=2.0];
  N16[label="*"];
  N17[label=-1.0];
  N18[label="**"];
  N19[label=ToReal];
  N20[label="[[-0.0,-8.836000051815063e-05],\\\\n[-8.836000051815063e-05,-0.0]]"];
  N21[label=MatrixScale];
  N22[label=MatrixExp];
  N23[label=MatrixScale];
  N24[label=ToRealMatrix];
  N25[label="[[0.0010000000474974513,0.0],\\\\n[0.0,0.0010000000474974513]]"];
  N26[label=MatrixAdd];
  N27[label=Cholesky];
  N28[label=2];
  N29[label=1];
  N30[label=ToMatrix];
  N31[label="@"];
  N32[label=0];
  N33[label=index];
  N34[label=Normal];
  N35[label=Sample];
  N36[label=index];
  N37[label=Normal];
  N38[label=Sample];
  N39[label="[4.0,0.0]"];
  N40[label=Phi];
  N41[label=Log];
  N42[label=Phi];
  N43[label=Log];
  N44[label=ToMatrix];
  N45[label=ToRealMatrix];
  N46[label=ElementwiseMult];
  N47[label="[29850.0,2016.0]"];
  N48[label=complement];
  N49[label=Log];
  N50[label=complement];
  N51[label=Log];
  N52[label=ToMatrix];
  N53[label=ToRealMatrix];
  N54[label=ElementwiseMult];
  N55[label=MatrixAdd];
  N56[label=MatrixSum];
  N57[label=ToNegReal];
  N58[label=Log1mexp];
  N59[label="-"];
  N60[label=ToReal];
  N61[label="+"];
  N62[label="Bernoulli(logits)"];
  N63[label=Sample];
  N64[label="Observation True"];
  N65[label=ToMatrix];
  N66[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N14;
  N02 -> N14;
  N03 -> N04;
  N04 -> N05;
  N05 -> N16;
  N05 -> N16;
  N06 -> N08;
  N06 -> N10;
  N07 -> N08;
  N07 -> N10;
  N07 -> N12;
  N08 -> N09;
  N09 -> N30;
  N10 -> N11;
  N11 -> N30;
  N12 -> N13;
  N13 -> N34;
  N13 -> N37;
  N14 -> N23;
  N15 -> N16;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
  N19 -> N21;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N26;
  N25 -> N26;
  N26 -> N27;
  N27 -> N31;
  N28 -> N30;
  N28 -> N44;
  N28 -> N52;
  N28 -> N65;
  N29 -> N30;
  N29 -> N36;
  N29 -> N44;
  N29 -> N52;
  N29 -> N65;
  N30 -> N31;
  N31 -> N33;
  N31 -> N36;
  N32 -> N33;
  N33 -> N34;
  N34 -> N35;
  N35 -> N40;
  N35 -> N65;
  N36 -> N37;
  N37 -> N38;
  N38 -> N42;
  N38 -> N65;
  N39 -> N46;
  N40 -> N41;
  N40 -> N48;
  N41 -> N44;
  N42 -> N43;
  N42 -> N50;
  N43 -> N44;
  N44 -> N45;
  N45 -> N46;
  N46 -> N55;
  N47 -> N54;
  N48 -> N49;
  N49 -> N52;
  N50 -> N51;
  N51 -> N52;
  N52 -> N53;
  N53 -> N54;
  N54 -> N55;
  N55 -> N56;
  N56 -> N57;
  N56 -> N61;
  N57 -> N58;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N61 -> N62;
  N62 -> N63;
  N63 -> N64;
  N65 -> N66;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
