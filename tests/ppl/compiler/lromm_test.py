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
  N16[label=FillMatrix];
  N17[label=FillMatrix];
  N18[label="[0,3,9]"];
  N19[label=MatrixScale];
  N20[label=MatrixAdd];
  N21[label=0];
  N22[label=index];
  N23[label=Normal];
  N24[label=33.0];
  N25[label=LogProb];
  N26[label=index];
  N27[label=Normal];
  N28[label=68.0];
  N29[label=LogProb];
  N30[label=2];
  N31[label=index];
  N32[label=Normal];
  N33[label=34.0];
  N34[label=LogProb];
  N35[label=ToMatrix];
  N36[label=MatrixAdd];
  N37[label=index];
  N38[label=complement];
  N39[label=Log];
  N40[label=ToReal];
  N41[label=FillMatrix];
  N42[label=3.5999999046325684];
  N43[label=Normal];
  N44[label=LogProb];
  N45[label=3.9000000953674316];
  N46[label=Normal];
  N47[label=LogProb];
  N48[label=2.5999999046325684];
  N49[label=Normal];
  N50[label=LogProb];
  N51[label=ToMatrix];
  N52[label=MatrixAdd];
  N53[label=index];
  N54[label=LogSumExp];
  N55[label=index];
  N56[label=index];
  N57[label=LogSumExp];
  N58[label=index];
  N59[label=index];
  N60[label=LogSumExp];
  N61[label=ToMatrix];
  N62[label=MatrixExp];
  N63[label=index];
  N64[label=ToProb];
  N65[label=Bernoulli];
  N66[label=Sample];
  N67[label=index];
  N68[label=ToProb];
  N69[label=Bernoulli];
  N70[label=Sample];
  N71[label=index];
  N72[label=ToProb];
  N73[label=Bernoulli];
  N74[label=Sample];
  N75[label="Observation True"];
  N76[label="Observation True"];
  N77[label="Observation True"];
  N78[label=Query];
  N79[label=Query];
  N80[label=Query];
  N81[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N14;
  N03 -> N38;
  N03 -> N81;
  N04 -> N06;
  N05 -> N06;
  N06 -> N07;
  N06 -> N08;
  N07 -> N17;
  N07 -> N78;
  N08 -> N19;
  N08 -> N79;
  N09 -> N10;
  N09 -> N10;
  N10 -> N11;
  N11 -> N23;
  N11 -> N27;
  N11 -> N32;
  N11 -> N80;
  N12 -> N16;
  N12 -> N17;
  N12 -> N35;
  N12 -> N41;
  N12 -> N51;
  N12 -> N61;
  N13 -> N16;
  N13 -> N17;
  N13 -> N26;
  N13 -> N35;
  N13 -> N41;
  N13 -> N51;
  N13 -> N55;
  N13 -> N56;
  N13 -> N61;
  N13 -> N67;
  N14 -> N15;
  N15 -> N16;
  N16 -> N36;
  N17 -> N20;
  N18 -> N19;
  N19 -> N20;
  N20 -> N22;
  N20 -> N26;
  N20 -> N31;
  N21 -> N22;
  N21 -> N37;
  N21 -> N53;
  N21 -> N63;
  N22 -> N23;
  N22 -> N43;
  N23 -> N25;
  N24 -> N25;
  N24 -> N44;
  N25 -> N35;
  N26 -> N27;
  N26 -> N46;
  N27 -> N29;
  N28 -> N29;
  N28 -> N47;
  N29 -> N35;
  N30 -> N31;
  N30 -> N58;
  N30 -> N59;
  N30 -> N71;
  N31 -> N32;
  N31 -> N49;
  N32 -> N34;
  N33 -> N34;
  N33 -> N50;
  N34 -> N35;
  N35 -> N36;
  N36 -> N37;
  N36 -> N55;
  N36 -> N58;
  N37 -> N54;
  N38 -> N39;
  N39 -> N40;
  N40 -> N41;
  N41 -> N52;
  N42 -> N43;
  N43 -> N44;
  N44 -> N51;
  N45 -> N46;
  N46 -> N47;
  N47 -> N51;
  N48 -> N49;
  N49 -> N50;
  N50 -> N51;
  N51 -> N52;
  N52 -> N53;
  N52 -> N56;
  N52 -> N59;
  N53 -> N54;
  N54 -> N61;
  N55 -> N57;
  N56 -> N57;
  N57 -> N61;
  N58 -> N60;
  N59 -> N60;
  N60 -> N61;
  N61 -> N62;
  N62 -> N63;
  N62 -> N67;
  N62 -> N71;
  N63 -> N64;
  N64 -> N65;
  N65 -> N66;
  N66 -> N75;
  N67 -> N68;
  N68 -> N69;
  N69 -> N70;
  N70 -> N76;
  N71 -> N72;
  N72 -> N73;
  N73 -> N74;
  N74 -> N77;
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
  N16[label="FillMatrix"];
  N17[label="FillMatrix"];
  N18[label="matrix"];
  N19[label="MatrixScale"];
  N20[label="MatrixAdd"];
  N21[label="0"];
  N22[label="Index"];
  N23[label="Normal"];
  N24[label="33"];
  N25[label="LogProb"];
  N26[label="Index"];
  N27[label="Normal"];
  N28[label="68"];
  N29[label="LogProb"];
  N30[label="2"];
  N31[label="Index"];
  N32[label="Normal"];
  N33[label="34"];
  N34[label="LogProb"];
  N35[label="ToMatrix"];
  N36[label="MatrixAdd"];
  N37[label="Index"];
  N38[label="Complement"];
  N39[label="Log"];
  N40[label="ToReal"];
  N41[label="FillMatrix"];
  N42[label="3.6"];
  N43[label="Normal"];
  N44[label="LogProb"];
  N45[label="3.9"];
  N46[label="Normal"];
  N47[label="LogProb"];
  N48[label="2.6"];
  N49[label="Normal"];
  N50[label="LogProb"];
  N51[label="ToMatrix"];
  N52[label="MatrixAdd"];
  N53[label="Index"];
  N54[label="LogSumExp"];
  N55[label="Index"];
  N56[label="Index"];
  N57[label="LogSumExp"];
  N58[label="Index"];
  N59[label="Index"];
  N60[label="LogSumExp"];
  N61[label="ToMatrix"];
  N62[label="1"];
  N63[label="FillMatrix"];
  N64[label="MatrixExp"];
  N65[label="MatrixNegate"];
  N66[label="ToReal"];
  N67[label="MatrixAdd"];
  N68[label="ToPosReal"];
  N69[label="MatrixLog"];
  N70[label="MatrixNegate"];
  N71[label="MatrixAdd"];
  N72[label="Index"];
  N73[label="BernoulliLogit"];
  N74[label="~"];
  N75[label="Index"];
  N76[label="BernoulliLogit"];
  N77[label="~"];
  N78[label="Index"];
  N79[label="BernoulliLogit"];
  N80[label="~"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N14;
  N3 -> N38;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N6 -> N8;
  N7 -> N17;
  N8 -> N19;
  N9 -> N10;
  N9 -> N10;
  N10 -> N11;
  N11 -> N23;
  N11 -> N27;
  N11 -> N32;
  N12 -> N16;
  N12 -> N17;
  N12 -> N35;
  N12 -> N41;
  N12 -> N51;
  N12 -> N61;
  N12 -> N63;
  N13 -> N16;
  N13 -> N17;
  N13 -> N26;
  N13 -> N35;
  N13 -> N41;
  N13 -> N51;
  N13 -> N55;
  N13 -> N56;
  N13 -> N61;
  N13 -> N63;
  N13 -> N75;
  N14 -> N15;
  N15 -> N16;
  N16 -> N36;
  N17 -> N20;
  N18 -> N19;
  N19 -> N20;
  N20 -> N22;
  N20 -> N26;
  N20 -> N31;
  N21 -> N22;
  N21 -> N37;
  N21 -> N53;
  N21 -> N72;
  N22 -> N23;
  N22 -> N43;
  N23 -> N25;
  N24 -> N25;
  N24 -> N44;
  N25 -> N35;
  N26 -> N27;
  N26 -> N46;
  N27 -> N29;
  N28 -> N29;
  N28 -> N47;
  N29 -> N35;
  N30 -> N31;
  N30 -> N58;
  N30 -> N59;
  N30 -> N78;
  N31 -> N32;
  N31 -> N49;
  N32 -> N34;
  N33 -> N34;
  N33 -> N50;
  N34 -> N35;
  N35 -> N36;
  N36 -> N37;
  N36 -> N55;
  N36 -> N58;
  N37 -> N54;
  N38 -> N39;
  N39 -> N40;
  N40 -> N41;
  N41 -> N52;
  N42 -> N43;
  N43 -> N44;
  N44 -> N51;
  N45 -> N46;
  N46 -> N47;
  N47 -> N51;
  N48 -> N49;
  N49 -> N50;
  N50 -> N51;
  N51 -> N52;
  N52 -> N53;
  N52 -> N56;
  N52 -> N59;
  N53 -> N54;
  N54 -> N61;
  N55 -> N57;
  N56 -> N57;
  N57 -> N61;
  N58 -> N60;
  N59 -> N60;
  N60 -> N61;
  N61 -> N64;
  N61 -> N71;
  N62 -> N63;
  N63 -> N67;
  N64 -> N65;
  N65 -> N66;
  N66 -> N67;
  N67 -> N68;
  N68 -> N69;
  N69 -> N70;
  N70 -> N71;
  N71 -> N72;
  N71 -> N75;
  N71 -> N78;
  N72 -> N73;
  N73 -> N74;
  N75 -> N76;
  N76 -> N77;
  N78 -> N79;
  N79 -> N80;
  O0[label="Observation"];
  N74 -> O0;
  O1[label="Observation"];
  N77 -> O1;
  O2[label="Observation"];
  N80 -> O2;
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
