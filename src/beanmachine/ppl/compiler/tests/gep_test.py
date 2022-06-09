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
    # TODO Fix this "hint" hack.
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
  N06[label=1.0];
  N07[label=HalfNormal];
  N08[label=Sample];
  N09[label=4.0];
  N10[label=2];
  N11[label="*"];
  N12[label=-0.0];
  N13[label=Exp];
  N14[label="8.836000051815063e-05"];
  N15[label=2.0];
  N16[label="*"];
  N17[label=-1.0];
  N18[label="**"];
  N19[label="*"];
  N20[label="-"];
  N21[label=Exp];
  N22[label=Exp];
  N23[label=Exp];
  N24[label=ToMatrix];
  N25[label=ToPosRealMatrix];
  N26[label=MatrixScale];
  N27[label=0];
  N28[label=ColumnIndex];
  N29[label=index];
  N30[label=0.0010000000474974513];
  N31[label="+"];
  N32[label=1];
  N33[label=index];
  N34[label=ColumnIndex];
  N35[label=index];
  N36[label=index];
  N37[label="+"];
  N38[label=ToMatrix];
  N39[label=Cholesky];
  N40[label=-0.0];
  N41[label=Normal];
  N42[label=Sample];
  N43[label=Normal];
  N44[label=Sample];
  N45[label=ToMatrix];
  N46[label="@"];
  N47[label=index];
  N48[label=Normal];
  N49[label=Sample];
  N50[label=Phi];
  N51[label=Log];
  N52[label="-"];
  N53[label="*"];
  N54[label="-"];
  N55[label=29850.0];
  N56[label=complement];
  N57[label=Log];
  N58[label="-"];
  N59[label="*"];
  N60[label="-"];
  N61[label="+"];
  N62[label=2016.0];
  N63[label=index];
  N64[label=Normal];
  N65[label=Sample];
  N66[label=Phi];
  N67[label=complement];
  N68[label=Log];
  N69[label="-"];
  N70[label="*"];
  N71[label="-"];
  N72[label="+"];
  N73[label=ToReal];
  N74[label=Log1mexp];
  N75[label="-"];
  N76[label=ToReal];
  N77[label="+"];
  N78[label="Bernoulli(logits)"];
  N79[label=Sample];
  N80[label="Observation True"];
  N81[label=ToMatrix];
  N82[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N11;
  N02 -> N11;
  N03 -> N04;
  N04 -> N05;
  N05 -> N16;
  N05 -> N16;
  N06 -> N07;
  N06 -> N41;
  N06 -> N43;
  N07 -> N08;
  N08 -> N48;
  N08 -> N64;
  N09 -> N53;
  N10 -> N24;
  N10 -> N24;
  N10 -> N38;
  N10 -> N38;
  N10 -> N45;
  N10 -> N81;
  N11 -> N26;
  N12 -> N13;
  N12 -> N23;
  N13 -> N24;
  N14 -> N19;
  N15 -> N16;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N20 -> N22;
  N21 -> N24;
  N22 -> N24;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N28;
  N26 -> N34;
  N27 -> N28;
  N27 -> N29;
  N27 -> N35;
  N27 -> N47;
  N28 -> N29;
  N28 -> N33;
  N29 -> N31;
  N30 -> N31;
  N30 -> N37;
  N31 -> N38;
  N32 -> N33;
  N32 -> N34;
  N32 -> N36;
  N32 -> N45;
  N32 -> N63;
  N32 -> N81;
  N33 -> N38;
  N34 -> N35;
  N34 -> N36;
  N35 -> N38;
  N36 -> N37;
  N37 -> N38;
  N38 -> N39;
  N39 -> N46;
  N40 -> N41;
  N40 -> N43;
  N41 -> N42;
  N42 -> N45;
  N43 -> N44;
  N44 -> N45;
  N45 -> N46;
  N46 -> N47;
  N46 -> N63;
  N47 -> N48;
  N48 -> N49;
  N49 -> N50;
  N49 -> N81;
  N50 -> N51;
  N50 -> N56;
  N51 -> N52;
  N52 -> N53;
  N53 -> N54;
  N54 -> N61;
  N55 -> N59;
  N56 -> N57;
  N57 -> N58;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N61 -> N72;
  N62 -> N70;
  N63 -> N64;
  N64 -> N65;
  N65 -> N66;
  N65 -> N81;
  N66 -> N67;
  N67 -> N68;
  N68 -> N69;
  N69 -> N70;
  N70 -> N71;
  N71 -> N72;
  N72 -> N73;
  N72 -> N74;
  N73 -> N77;
  N74 -> N75;
  N75 -> N76;
  N76 -> N77;
  N77 -> N78;
  N78 -> N79;
  N79 -> N80;
  N81 -> N82;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
