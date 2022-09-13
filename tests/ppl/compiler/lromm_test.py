# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compilation test of Todd's Linear Regression Outliers Marginalized model"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.distributions.unit import Unit
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import logaddexp, ones, tensor
from torch.distributions import Bernoulli, Beta, Gamma, Normal


_x_obs = tensor([0, 3, 9])
_y_obs = tensor([33, 68, 34])
_err_obs = tensor([3.6, 3.9, 2.6])


@bm.random_variable
def beta_0():
    return Normal(0, 10)


@bm.random_variable
def beta_1():
    return Normal(0, 10)


@bm.random_variable
def sigma_out():
    return Gamma(1, 1)


@bm.random_variable
def theta():
    return Beta(2, 5)


@bm.functional
def f():
    mu = beta_0() + beta_1() * _x_obs
    ns = Normal(mu, sigma_out())
    ne = Normal(mu, _err_obs)
    log_likelihood_outlier = theta().log() + ns.log_prob(_y_obs)
    log_likelihood = (1 - theta()).log() + ne.log_prob(_y_obs)
    return logaddexp(log_likelihood_outlier, log_likelihood)


@bm.random_variable
def y():
    return Unit(f())


# Same model, but with the "Bernoulli trick" instead of a Unit:


@bm.random_variable
def d():
    return Bernoulli(f().exp())


class LROMMTest(unittest.TestCase):
    def test_lromm_unit_to_dot(self) -> None:
        self.maxDiff = None
        queries = [beta_0(), beta_1(), sigma_out(), theta()]
        observations = {y(): _y_obs}
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot(queries, observations)
        expected = """
Function Unit is not supported by Bean Machine Graph.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_lromm_bern_to_dot(self) -> None:
        self.maxDiff = None
        queries = [beta_0(), beta_1(), sigma_out(), theta()]
        observations = {d(): ones(len(_y_obs))}
        observed = BMGInference().to_dot(queries, observations)
        # TODO: are ToMatrix and ToRealMatrix needed?
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=5.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=0.0];
  N05[label=10.0];
  N06[label=Normal];
  N07[label=Sample];
  N08[label=Sample];
  N09[label=1.0];
  N10[label=Gamma];
  N11[label=Sample];
  N12[label=3];
  N13[label=1];
  N14[label=Log];
  N15[label=ToReal];
  N16[label="[0,3,9]"];
  N17[label=MatrixScale];
  N18[label=0];
  N19[label=index];
  N20[label="+"];
  N21[label=Normal];
  N22[label=33.0];
  N23[label=LogProb];
  N24[label="+"];
  N25[label=complement];
  N26[label=Log];
  N27[label=ToReal];
  N28[label=3.5999999046325684];
  N29[label=Normal];
  N30[label=LogProb];
  N31[label="+"];
  N32[label=LogSumExp];
  N33[label=index];
  N34[label="+"];
  N35[label=Normal];
  N36[label=68.0];
  N37[label=LogProb];
  N38[label="+"];
  N39[label=3.9000000953674316];
  N40[label=Normal];
  N41[label=LogProb];
  N42[label="+"];
  N43[label=LogSumExp];
  N44[label=2];
  N45[label=index];
  N46[label="+"];
  N47[label=Normal];
  N48[label=34.0];
  N49[label=LogProb];
  N50[label="+"];
  N51[label=2.5999999046325684];
  N52[label=Normal];
  N53[label=LogProb];
  N54[label="+"];
  N55[label=LogSumExp];
  N56[label=ToMatrix];
  N57[label=MatrixExp];
  N58[label=index];
  N59[label=ToProb];
  N60[label=Bernoulli];
  N61[label=Sample];
  N62[label=index];
  N63[label=ToProb];
  N64[label=Bernoulli];
  N65[label=Sample];
  N66[label=index];
  N67[label=ToProb];
  N68[label=Bernoulli];
  N69[label=Sample];
  N70[label="Observation True"];
  N71[label="Observation True"];
  N72[label="Observation True"];
  N73[label=Query];
  N74[label=Query];
  N75[label=Query];
  N76[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N14;
  N03 -> N25;
  N03 -> N76;
  N04 -> N06;
  N05 -> N06;
  N06 -> N07;
  N06 -> N08;
  N07 -> N20;
  N07 -> N34;
  N07 -> N46;
  N07 -> N73;
  N08 -> N17;
  N08 -> N74;
  N09 -> N10;
  N09 -> N10;
  N10 -> N11;
  N11 -> N21;
  N11 -> N35;
  N11 -> N47;
  N11 -> N75;
  N12 -> N56;
  N13 -> N33;
  N13 -> N56;
  N13 -> N62;
  N14 -> N15;
  N15 -> N24;
  N15 -> N38;
  N15 -> N50;
  N16 -> N17;
  N17 -> N19;
  N17 -> N33;
  N17 -> N45;
  N18 -> N19;
  N18 -> N58;
  N19 -> N20;
  N20 -> N21;
  N20 -> N29;
  N21 -> N23;
  N22 -> N23;
  N22 -> N30;
  N23 -> N24;
  N24 -> N32;
  N25 -> N26;
  N26 -> N27;
  N27 -> N31;
  N27 -> N42;
  N27 -> N54;
  N28 -> N29;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
  N32 -> N56;
  N33 -> N34;
  N34 -> N35;
  N34 -> N40;
  N35 -> N37;
  N36 -> N37;
  N36 -> N41;
  N37 -> N38;
  N38 -> N43;
  N39 -> N40;
  N40 -> N41;
  N41 -> N42;
  N42 -> N43;
  N43 -> N56;
  N44 -> N45;
  N44 -> N66;
  N45 -> N46;
  N46 -> N47;
  N46 -> N52;
  N47 -> N49;
  N48 -> N49;
  N48 -> N53;
  N49 -> N50;
  N50 -> N55;
  N51 -> N52;
  N52 -> N53;
  N53 -> N54;
  N54 -> N55;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N57 -> N62;
  N57 -> N66;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N61 -> N70;
  N62 -> N63;
  N63 -> N64;
  N64 -> N65;
  N65 -> N71;
  N66 -> N67;
  N67 -> N68;
  N68 -> N69;
  N69 -> N72;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
