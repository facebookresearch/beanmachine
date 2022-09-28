# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import (
    Bernoulli,
    Beta,
    Gamma,
    HalfCauchy,
    Normal,
    StudentT,
    Uniform,
)


@bm.random_variable
def beta(n):
    return Beta(2.0, 2.0)


@bm.random_variable
def flip_beta():
    return Bernoulli(tensor([beta(0), beta(1)]))


@bm.random_variable
def beta_2_2():
    return Beta(2.0, tensor([3.0, 4.0]))


@bm.random_variable
def flip_beta_2_2():
    return Bernoulli(beta_2_2())


@bm.random_variable
def uniform_2_2():
    return Uniform(0.0, tensor([1.0, 1.0]))


@bm.random_variable
def flip_uniform_2_2():
    return Bernoulli(uniform_2_2())


@bm.random_variable
def flip_logits():
    return Bernoulli(logits=tensor([beta(0), beta(1)]))


@bm.random_variable
def flip_const():
    return Bernoulli(tensor([0.25, 0.75]))


@bm.random_variable
def flip_const_4():
    return Bernoulli(tensor([0.25, 0.75, 0.5, 0.5]))


@bm.random_variable
def flip_const_2_3():
    return Bernoulli(tensor([[0.25, 0.75, 0.5], [0.125, 0.875, 0.625]]))


@bm.random_variable
def normal_2_3():
    mus = flip_const_2_3()  # 2 x 3 tensor of 0 or 1
    sigmas = tensor([2.0, 3.0, 4.0])

    return Normal(mus, sigmas)


@bm.random_variable
def hc_3():
    return HalfCauchy(tensor([1.0, 2.0, 3.0]))


@bm.random_variable
def studentt_2_3():
    return StudentT(hc_3(), normal_2_3(), hc_3())


@bm.functional
def operators():
    # Note that we do NOT devectorize the multiplication; it gets
    # turned into a MatrixScale.
    return ((beta_2_2() + tensor([[5.0, 6.0], [7.0, 8.0]])) * 10.0).exp()


@bm.functional
def multiplication():
    return beta_2_2() * tensor([5.0, 6.0])


@bm.functional
def complement_with_log1p():
    return (-beta_2_2()).log1p()


@bm.random_variable
def beta1234():
    return Beta(tensor([1.0, 2.0]), tensor([3.0, 4.0]))


@bm.functional
def sum_inverted_log_probs():
    p = tensor([5.0, 6.0]) * (-beta1234()).log1p()
    return p.sum()


@bm.random_variable
def gamma():
    return Gamma(1, 1)


@bm.functional
def normal_log_probs():
    mu = tensor([5.0, 6.0])
    x = tensor([7.0, 8.0])
    return Normal(mu, gamma()).log_prob(x)


class FixVectorizedModelsTest(unittest.TestCase):
    def test_fix_vectorized_models_1(self) -> None:
        self.maxDiff = None
        observations = {flip_beta(): tensor([0.0, 1.0])}
        queries = [flip_beta(), flip_const()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=Tensor];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label="Observation tensor([0., 1.])"];
  N08[label=Query];
  N09[label="[0.25,0.75]"];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N04;
  N03 -> N04;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N06 -> N08;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label="Observation False"];
  N09[label="Observation True"];
  N10[label=2];
  N11[label=1];
  N12[label=ToMatrix];
  N13[label=Query];
  N14[label=0.25];
  N15[label=Bernoulli];
  N16[label=Sample];
  N17[label=0.75];
  N18[label=Bernoulli];
  N19[label=Sample];
  N20[label=ToMatrix];
  N21[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N04;
  N03 -> N06;
  N04 -> N05;
  N05 -> N08;
  N05 -> N12;
  N06 -> N07;
  N07 -> N09;
  N07 -> N12;
  N10 -> N12;
  N10 -> N20;
  N11 -> N12;
  N11 -> N20;
  N12 -> N13;
  N14 -> N15;
  N15 -> N16;
  N16 -> N20;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_2(self) -> None:
        self.maxDiff = None
        observations = {flip_const_4(): tensor([0.0, 1.0, 0.0, 1.0])}
        queries = [flip_const_4()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N0[label="[0.25,0.75,0.5,0.5]"];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label="Observation tensor([0., 1., 0., 1.])"];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N2 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        # Note that due to the order in which we do the rewriting we
        # end up with a not-deduplicated Bernoulli(0.5) node here, which
        # is slightly unfortunate but probably not worth fixing right now.

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=0.25];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.75];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=0.5];
  N07[label=Bernoulli];
  N08[label=Sample];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label="Observation False"];
  N12[label="Observation True"];
  N13[label="Observation False"];
  N14[label="Observation True"];
  N15[label=4];
  N16[label=1];
  N17[label=ToMatrix];
  N18[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N11;
  N02 -> N17;
  N03 -> N04;
  N04 -> N05;
  N05 -> N12;
  N05 -> N17;
  N06 -> N07;
  N06 -> N09;
  N07 -> N08;
  N08 -> N13;
  N08 -> N17;
  N09 -> N10;
  N10 -> N14;
  N10 -> N17;
  N15 -> N17;
  N16 -> N17;
  N17 -> N18;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_3(self) -> None:
        self.maxDiff = None
        observations = {flip_const_2_3(): tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])}
        queries = [flip_const_2_3()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N0[label="[[0.25,0.75,0.5],\\\\n[0.125,0.875,0.625]]"];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label="Observation tensor([[0., 0., 0.],\\n        [1., 1., 1.]])"];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N2 -> N4;
}
    """
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=0.25];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.75];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=0.5];
  N07[label=Bernoulli];
  N08[label=Sample];
  N09[label=0.125];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=0.875];
  N13[label=Bernoulli];
  N14[label=Sample];
  N15[label=0.625];
  N16[label=Bernoulli];
  N17[label=Sample];
  N18[label="Observation False"];
  N19[label="Observation False"];
  N20[label="Observation False"];
  N21[label="Observation True"];
  N22[label="Observation True"];
  N23[label="Observation True"];
  N24[label=3];
  N25[label=2];
  N26[label=ToMatrix];
  N27[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N18;
  N02 -> N26;
  N03 -> N04;
  N04 -> N05;
  N05 -> N19;
  N05 -> N26;
  N06 -> N07;
  N07 -> N08;
  N08 -> N20;
  N08 -> N26;
  N09 -> N10;
  N10 -> N11;
  N11 -> N21;
  N11 -> N26;
  N12 -> N13;
  N13 -> N14;
  N14 -> N22;
  N14 -> N26;
  N15 -> N16;
  N16 -> N17;
  N17 -> N23;
  N17 -> N26;
  N24 -> N26;
  N25 -> N26;
  N26 -> N27;
}
    """
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_4(self) -> None:

        # Demonstrate we can also do devectorizations on logits-style Bernoullis.
        # (A logits Bernoulli with a beta prior is a likely mistake in a real model,
        # but it is a convenient test case.)

        self.maxDiff = None
        observations = {}
        queries = [flip_logits()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=Tensor];
  N5[label="Bernoulli(logits)"];
  N6[label=Sample];
  N7[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N4;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
  N6 -> N7;
}
    """
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=ToReal];
  N05[label="Bernoulli(logits)"];
  N06[label=Sample];
  N07[label=ToReal];
  N08[label="Bernoulli(logits)"];
  N09[label=Sample];
  N10[label=2];
  N11[label=1];
  N12[label=ToMatrix];
  N13[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N04;
  N03 -> N07;
  N04 -> N05;
  N05 -> N06;
  N06 -> N12;
  N07 -> N08;
  N08 -> N09;
  N09 -> N12;
  N10 -> N12;
  N11 -> N12;
  N12 -> N13;
}
    """
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_5(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [studentt_2_3()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite. Note that we have a size[3] stochastic input and
        # a size[2, 3] stochastic input to the StudentT, and we broadcast the three
        # HalfCauchy samples correctly

        expected = """
digraph "graph" {
  N00[label="[1.0,2.0,3.0]"];
  N01[label=HalfCauchy];
  N02[label=Sample];
  N03[label="[[0.25,0.75,0.5],\\\\n[0.125,0.875,0.625]]"];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label="[2.0,3.0,4.0]"];
  N07[label=Normal];
  N08[label=Sample];
  N09[label=StudentT];
  N10[label=Sample];
  N11[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N09;
  N02 -> N09;
  N03 -> N04;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=HalfCauchy];
  N02[label=Sample];
  N03[label=2.0];
  N04[label=HalfCauchy];
  N05[label=Sample];
  N06[label=3.0];
  N07[label=HalfCauchy];
  N08[label=Sample];
  N09[label=0.25];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=0.75];
  N13[label=Bernoulli];
  N14[label=Sample];
  N15[label=0.5];
  N16[label=Bernoulli];
  N17[label=Sample];
  N18[label=0.125];
  N19[label=Bernoulli];
  N20[label=Sample];
  N21[label=0.875];
  N22[label=Bernoulli];
  N23[label=Sample];
  N24[label=0.625];
  N25[label=Bernoulli];
  N26[label=Sample];
  N27[label=ToReal];
  N28[label=Normal];
  N29[label=Sample];
  N30[label=ToReal];
  N31[label=Normal];
  N32[label=Sample];
  N33[label=ToReal];
  N34[label=4.0];
  N35[label=Normal];
  N36[label=Sample];
  N37[label=ToReal];
  N38[label=Normal];
  N39[label=Sample];
  N40[label=ToReal];
  N41[label=Normal];
  N42[label=Sample];
  N43[label=ToReal];
  N44[label=Normal];
  N45[label=Sample];
  N46[label=StudentT];
  N47[label=Sample];
  N48[label=StudentT];
  N49[label=Sample];
  N50[label=StudentT];
  N51[label=Sample];
  N52[label=StudentT];
  N53[label=Sample];
  N54[label=StudentT];
  N55[label=Sample];
  N56[label=StudentT];
  N57[label=Sample];
  N58[label=3];
  N59[label=2];
  N60[label=ToMatrix];
  N61[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N46;
  N02 -> N46;
  N02 -> N52;
  N02 -> N52;
  N03 -> N04;
  N03 -> N28;
  N03 -> N38;
  N04 -> N05;
  N05 -> N48;
  N05 -> N48;
  N05 -> N54;
  N05 -> N54;
  N06 -> N07;
  N06 -> N31;
  N06 -> N41;
  N07 -> N08;
  N08 -> N50;
  N08 -> N50;
  N08 -> N56;
  N08 -> N56;
  N09 -> N10;
  N10 -> N11;
  N11 -> N27;
  N12 -> N13;
  N13 -> N14;
  N14 -> N30;
  N15 -> N16;
  N16 -> N17;
  N17 -> N33;
  N18 -> N19;
  N19 -> N20;
  N20 -> N37;
  N21 -> N22;
  N22 -> N23;
  N23 -> N40;
  N24 -> N25;
  N25 -> N26;
  N26 -> N43;
  N27 -> N28;
  N28 -> N29;
  N29 -> N46;
  N30 -> N31;
  N31 -> N32;
  N32 -> N48;
  N33 -> N35;
  N34 -> N35;
  N34 -> N44;
  N35 -> N36;
  N36 -> N50;
  N37 -> N38;
  N38 -> N39;
  N39 -> N52;
  N40 -> N41;
  N41 -> N42;
  N42 -> N54;
  N43 -> N44;
  N44 -> N45;
  N45 -> N56;
  N46 -> N47;
  N47 -> N60;
  N48 -> N49;
  N49 -> N60;
  N50 -> N51;
  N51 -> N60;
  N52 -> N53;
  N53 -> N60;
  N54 -> N55;
  N55 -> N60;
  N56 -> N57;
  N57 -> N60;
  N58 -> N60;
  N59 -> N60;
  N60 -> N61;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_6(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [flip_beta_2_2(), flip_uniform_2_2()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite: notice that here torch automatically
        # broadcast the 2.0 to [2.0, 2.0] for us when the node was accumulated,
        # and similarly for 0.0.

        expected = """
digraph "graph" {
  N00[label="[2.0,2.0]"];
  N01[label="[3.0,4.0]"];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=Query];
  N07[label="[0.0,0.0]"];
  N08[label="[1.0,1.0]"];
  N09[label=Uniform];
  N10[label=Sample];
  N11[label=Bernoulli];
  N12[label=Sample];
  N13[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N04 -> N05;
  N05 -> N06;
  N07 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After: notice that we correctly generate two samples from a Flat distribution
        # here.

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=3.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=4.0];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=Bernoulli];
  N08[label=Sample];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=2];
  N12[label=1];
  N13[label=ToMatrix];
  N14[label=Query];
  N15[label=Flat];
  N16[label=Sample];
  N17[label=Sample];
  N18[label=Bernoulli];
  N19[label=Sample];
  N20[label=Bernoulli];
  N21[label=Sample];
  N22[label=ToMatrix];
  N23[label=Query];
  N00 -> N02;
  N00 -> N05;
  N01 -> N02;
  N02 -> N03;
  N03 -> N07;
  N04 -> N05;
  N05 -> N06;
  N06 -> N09;
  N07 -> N08;
  N08 -> N13;
  N09 -> N10;
  N10 -> N13;
  N11 -> N13;
  N11 -> N22;
  N12 -> N13;
  N12 -> N22;
  N13 -> N14;
  N15 -> N16;
  N15 -> N17;
  N16 -> N18;
  N17 -> N20;
  N18 -> N19;
  N19 -> N22;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
}

"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_7(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [operators()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N0[label="[2.0,2.0]"];
  N1[label="[3.0,4.0]"];
  N2[label=Beta];
  N3[label=Sample];
  N4[label="[[5.0,6.0],\\\\n[7.0,8.0]]"];
  N5[label="+"];
  N6[label=10.0];
  N7[label="*"];
  N8[label=Exp];
  N9[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N7;
  N6 -> N7;
  N7 -> N8;
  N8 -> N9;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=3.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=4.0];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=10.0];
  N08[label=2];
  N09[label=1];
  N10[label=ToMatrix];
  N11[label=ToRealMatrix];
  N12[label="[[5.0,6.0],\\\\n[7.0,8.0]]"];
  N13[label=MatrixAdd];
  N14[label=MatrixScale];
  N15[label=MatrixExp];
  N16[label=Query];
  N00 -> N02;
  N00 -> N05;
  N01 -> N02;
  N02 -> N03;
  N03 -> N10;
  N04 -> N05;
  N05 -> N06;
  N06 -> N10;
  N07 -> N14;
  N08 -> N10;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_8(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [multiplication()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N0[label="[2.0,2.0]"];
  N1[label="[3.0,4.0]"];
  N2[label=Beta];
  N3[label=Sample];
  N4[label="[5.0,6.0]"];
  N5[label="*"];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=3.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=4.0];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=2];
  N08[label=1];
  N09[label=ToMatrix];
  N10[label=ToRealMatrix];
  N11[label="[5.0,6.0]"];
  N12[label=ElementwiseMult];
  N13[label=Query];
  N00 -> N02;
  N00 -> N05;
  N01 -> N02;
  N02 -> N03;
  N03 -> N09;
  N04 -> N05;
  N05 -> N06;
  N06 -> N09;
  N07 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N12;
  N11 -> N12;
  N12 -> N13;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_9(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [complement_with_log1p()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N0[label="[2.0,2.0]"];
  N1[label="[3.0,4.0]"];
  N2[label=Beta];
  N3[label=Sample];
  N4[label="-"];
  N5[label=Log1p];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=3.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=4.0];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=2];
  N08[label=1];
  N09[label=complement];
  N10[label=Log];
  N11[label=complement];
  N12[label=Log];
  N13[label=ToMatrix];
  N14[label=Query];
  N00 -> N02;
  N00 -> N05;
  N01 -> N02;
  N02 -> N03;
  N03 -> N09;
  N04 -> N05;
  N05 -> N06;
  N06 -> N11;
  N07 -> N13;
  N08 -> N13;
  N09 -> N10;
  N10 -> N13;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_10(self) -> None:
        self.maxDiff = None
        queries = [sum_inverted_log_probs()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=3.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=2.0];
  N05[label=4.0];
  N06[label=Beta];
  N07[label=Sample];
  N08[label="[5.0,6.0]"];
  N09[label=2];
  N10[label=1];
  N11[label=complement];
  N12[label=Log];
  N13[label=complement];
  N14[label=Log];
  N15[label=ToMatrix];
  N16[label=ToRealMatrix];
  N17[label=ElementwiseMult];
  N18[label=MatrixSum];
  N19[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N11;
  N04 -> N06;
  N05 -> N06;
  N06 -> N07;
  N07 -> N13;
  N08 -> N17;
  N09 -> N15;
  N10 -> N15;
  N11 -> N12;
  N12 -> N15;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_vectorized_models_11(self) -> None:
        self.maxDiff = None
        queries = [normal_log_probs()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=Gamma];
  N02[label=Sample];
  N03[label=2];
  N04[label=1];
  N05[label=5.0];
  N06[label=Normal];
  N07[label=7.0];
  N08[label=LogProb];
  N09[label=6.0];
  N10[label=Normal];
  N11[label=8.0];
  N12[label=LogProb];
  N13[label=ToMatrix];
  N14[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N06;
  N02 -> N10;
  N03 -> N13;
  N04 -> N13;
  N05 -> N06;
  N06 -> N08;
  N07 -> N08;
  N08 -> N13;
  N09 -> N10;
  N10 -> N12;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
}
        """
        self.assertEqual(observed.strip(), expected.strip())
