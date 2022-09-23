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


# Same model, but using a logits Bernoulli


@bm.random_variable
def d2():
    log_prob = f()
    logit = log_prob - (1 - log_prob.exp()).log()
    return Bernoulli(logits=logit)


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

    def test_lromm_logits_to_bmg_dot(self) -> None:
        self.maxDiff = None
        queries = [beta_0(), beta_1(), sigma_out(), theta()]
        observations = {d2(): ones(len(_y_obs))}
        # Go all the way to BMG.
        # (This regression-tests the bug described t131976521.)
        g, _ = BMGInference().to_graph(queries, observations)
        observed = g.to_dot()
        expected = """
digraph "graph" {
  N0[label="2"];
  N1[label="5"];
  N2[label="Beta"];
  N3[label="~"];
  N4[label="0"];
  N5[label="10"];
  N6[label="Normal"];
  N7[label="~"];
  N8[label="~"];
  N9[label="1"];
  N10[label="Gamma"];
  N11[label="~"];
  N12[label="3"];
  N13[label="1"];
  N14[label="Log"];
  N15[label="ToReal"];
  N16[label="matrix"];
  N17[label="MatrixScale"];
  N18[label="0"];
  N19[label="Index"];
  N20[label="+"];
  N21[label="Normal"];
  N22[label="33"];
  N23[label="LogProb"];
  N24[label="+"];
  N25[label="Complement"];
  N26[label="Log"];
  N27[label="ToReal"];
  N28[label="3.6"];
  N29[label="Normal"];
  N30[label="LogProb"];
  N31[label="+"];
  N32[label="LogSumExp"];
  N33[label="Index"];
  N34[label="+"];
  N35[label="Normal"];
  N36[label="68"];
  N37[label="LogProb"];
  N38[label="+"];
  N39[label="3.9"];
  N40[label="Normal"];
  N41[label="LogProb"];
  N42[label="+"];
  N43[label="LogSumExp"];
  N44[label="2"];
  N45[label="Index"];
  N46[label="+"];
  N47[label="Normal"];
  N48[label="34"];
  N49[label="LogProb"];
  N50[label="+"];
  N51[label="2.6"];
  N52[label="Normal"];
  N53[label="LogProb"];
  N54[label="+"];
  N55[label="LogSumExp"];
  N56[label="ToMatrix"];
  N57[label="1"];
  N58[label="MatrixExp"];
  N59[label="Index"];
  N60[label="Negate"];
  N61[label="ToReal"];
  N62[label="+"];
  N63[label="Index"];
  N64[label="Negate"];
  N65[label="ToReal"];
  N66[label="+"];
  N67[label="Index"];
  N68[label="Negate"];
  N69[label="ToReal"];
  N70[label="+"];
  N71[label="ToMatrix"];
  N72[label="ToPosReal"];
  N73[label="MatrixLog"];
  N74[label="Index"];
  N75[label="Negate"];
  N76[label="Index"];
  N77[label="Negate"];
  N78[label="Index"];
  N79[label="Negate"];
  N80[label="ToMatrix"];
  N81[label="MatrixAdd"];
  N82[label="Index"];
  N83[label="BernoulliLogit"];
  N84[label="~"];
  N85[label="Index"];
  N86[label="BernoulliLogit"];
  N87[label="~"];
  N88[label="Index"];
  N89[label="BernoulliLogit"];
  N90[label="~"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N14;
  N3 -> N25;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N6 -> N8;
  N7 -> N20;
  N7 -> N34;
  N7 -> N46;
  N8 -> N17;
  N9 -> N10;
  N9 -> N10;
  N10 -> N11;
  N11 -> N21;
  N11 -> N35;
  N11 -> N47;
  N12 -> N56;
  N12 -> N71;
  N12 -> N80;
  N13 -> N33;
  N13 -> N56;
  N13 -> N63;
  N13 -> N71;
  N13 -> N76;
  N13 -> N80;
  N13 -> N85;
  N14 -> N15;
  N15 -> N24;
  N15 -> N38;
  N15 -> N50;
  N16 -> N17;
  N17 -> N19;
  N17 -> N33;
  N17 -> N45;
  N18 -> N19;
  N18 -> N59;
  N18 -> N74;
  N18 -> N82;
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
  N44 -> N67;
  N44 -> N78;
  N44 -> N88;
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
  N56 -> N58;
  N56 -> N81;
  N57 -> N62;
  N57 -> N66;
  N57 -> N70;
  N58 -> N59;
  N58 -> N63;
  N58 -> N67;
  N59 -> N60;
  N60 -> N61;
  N61 -> N62;
  N62 -> N71;
  N63 -> N64;
  N64 -> N65;
  N65 -> N66;
  N66 -> N71;
  N67 -> N68;
  N68 -> N69;
  N69 -> N70;
  N70 -> N71;
  N71 -> N72;
  N72 -> N73;
  N73 -> N74;
  N73 -> N76;
  N73 -> N78;
  N74 -> N75;
  N75 -> N80;
  N76 -> N77;
  N77 -> N80;
  N78 -> N79;
  N79 -> N80;
  N80 -> N81;
  N81 -> N82;
  N81 -> N85;
  N81 -> N88;
  N82 -> N83;
  N83 -> N84;
  N85 -> N86;
  N86 -> N87;
  N88 -> N89;
  N89 -> N90;
  O0[label="Observation"];
  N84 -> O0;
  O1[label="Observation"];
  N87 -> O1;
  O2[label="Observation"];
  N90 -> O2;
  Q0[label="Query"];
  N7 -> Q0;
  Q1[label="Query"];
  N8 -> Q1;
  Q2[label="Query"];
  N11 -> Q2;
  Q3[label="Query"];
  N3 -> Q3;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
