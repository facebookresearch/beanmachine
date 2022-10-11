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
        # Demonstrate that compiling to an actual BMG graph
        # generates a graph which type checks.
        g, _ = BMGInference().to_graph(queries, observations)
        observed = g.to_dot()

        expected = """
digraph "graph" {
  N0[label="5"];
  N1[label="HalfNormal"];
  N2[label="~"];
  N3[label="0.1"];
  N4[label="HalfNormal"];
  N5[label="~"];
  N6[label="0"];
  N7[label="1"];
  N8[label="Normal"];
  N9[label="~"];
  N10[label="Normal"];
  N11[label="~"];
  N12[label="HalfNormal"];
  N13[label="~"];
  N14[label="*"];
  N15[label="2"];
  N16[label="*"];
  N17[label="-1"];
  N18[label="^"];
  N19[label="ToReal"];
  N20[label="matrix"];
  N21[label="MatrixScale"];
  N22[label="MatrixExp"];
  N23[label="MatrixScale"];
  N24[label="matrix"];
  N25[label="MatrixAdd"];
  N26[label="Cholesky"];
  N27[label="2"];
  N28[label="1"];
  N29[label="ToMatrix"];
  N30[label="MatrixMultiply"];
  N31[label="0"];
  N32[label="Index"];
  N33[label="Normal"];
  N34[label="~"];
  N35[label="Index"];
  N36[label="Normal"];
  N37[label="~"];
  N38[label="matrix"];
  N39[label="Phi"];
  N40[label="Phi"];
  N41[label="ToMatrix"];
  N42[label="MatrixLog"];
  N43[label="ToReal"];
  N44[label="ElementwiseMultiply"];
  N45[label="matrix"];
  N46[label="Complement"];
  N47[label="Log"];
  N48[label="Complement"];
  N49[label="Log"];
  N50[label="ToMatrix"];
  N51[label="ToReal"];
  N52[label="ElementwiseMultiply"];
  N53[label="MatrixAdd"];
  N54[label="MatrixSum"];
  N55[label="ToNegReal"];
  N56[label="Log1mExp"];
  N57[label="Negate"];
  N58[label="ToReal"];
  N59[label="+"];
  N60[label="BernoulliLogit"];
  N61[label="~"];
  N62[label="ToMatrix"];
  N0 -> N1;
  N1 -> N2;
  N2 -> N14;
  N2 -> N14;
  N3 -> N4;
  N4 -> N5;
  N5 -> N16;
  N5 -> N16;
  N6 -> N8;
  N6 -> N10;
  N7 -> N8;
  N7 -> N10;
  N7 -> N12;
  N8 -> N9;
  N9 -> N29;
  N10 -> N11;
  N11 -> N29;
  N12 -> N13;
  N13 -> N33;
  N13 -> N36;
  N14 -> N23;
  N15 -> N16;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
  N19 -> N21;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N23 -> N25;
  N24 -> N25;
  N25 -> N26;
  N26 -> N30;
  N27 -> N29;
  N27 -> N41;
  N27 -> N50;
  N27 -> N62;
  N28 -> N29;
  N28 -> N35;
  N28 -> N41;
  N28 -> N50;
  N28 -> N62;
  N29 -> N30;
  N30 -> N32;
  N30 -> N35;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
  N34 -> N39;
  N34 -> N62;
  N35 -> N36;
  N36 -> N37;
  N37 -> N40;
  N37 -> N62;
  N38 -> N44;
  N39 -> N41;
  N39 -> N46;
  N40 -> N41;
  N40 -> N48;
  N41 -> N42;
  N42 -> N43;
  N43 -> N44;
  N44 -> N53;
  N45 -> N52;
  N46 -> N47;
  N47 -> N50;
  N48 -> N49;
  N49 -> N50;
  N50 -> N51;
  N51 -> N52;
  N52 -> N53;
  N53 -> N54;
  N54 -> N55;
  N54 -> N59;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  O0[label="Observation"];
  N61 -> O0;
  Q0[label="Query"];
  N62 -> Q0;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
