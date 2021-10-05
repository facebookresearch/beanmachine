# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta, Normal, Uniform, HalfCauchy, StudentT


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
    return beta_2_2() + tensor([[5.0, 6.0], [7.0, 8.0]])


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
  N04[label=2];
  N05[label=1];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label=ToMatrix];
  N11[label=Query];
  N12[label=0.25];
  N13[label=Bernoulli];
  N14[label=Sample];
  N15[label=0.75];
  N16[label=Bernoulli];
  N17[label=Sample];
  N18[label=ToMatrix];
  N19[label=Query];
  N20[label="Observation False"];
  N21[label="Observation True"];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N06;
  N03 -> N08;
  N04 -> N10;
  N04 -> N18;
  N05 -> N10;
  N05 -> N18;
  N06 -> N07;
  N07 -> N10;
  N07 -> N20;
  N08 -> N09;
  N09 -> N10;
  N09 -> N21;
  N10 -> N11;
  N12 -> N13;
  N13 -> N14;
  N14 -> N18;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
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
  N00[label=4];
  N01[label=1];
  N02[label=0.25];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label=0.75];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label=0.5];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=Bernoulli];
  N12[label=Sample];
  N13[label=ToMatrix];
  N14[label=Query];
  N15[label="Observation False"];
  N16[label="Observation True"];
  N17[label="Observation False"];
  N18[label="Observation True"];
  N00 -> N13;
  N01 -> N13;
  N02 -> N03;
  N03 -> N04;
  N04 -> N13;
  N04 -> N15;
  N05 -> N06;
  N06 -> N07;
  N07 -> N13;
  N07 -> N16;
  N08 -> N09;
  N08 -> N11;
  N09 -> N10;
  N10 -> N13;
  N10 -> N17;
  N11 -> N12;
  N12 -> N13;
  N12 -> N18;
  N13 -> N14;
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
  N00[label=3];
  N01[label=2];
  N02[label=0.25];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label=0.75];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label=0.5];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=0.125];
  N12[label=Bernoulli];
  N13[label=Sample];
  N14[label=0.875];
  N15[label=Bernoulli];
  N16[label=Sample];
  N17[label=0.625];
  N18[label=Bernoulli];
  N19[label=Sample];
  N20[label=ToMatrix];
  N21[label=Query];
  N22[label="Observation False"];
  N23[label="Observation False"];
  N24[label="Observation False"];
  N25[label="Observation True"];
  N26[label="Observation True"];
  N27[label="Observation True"];
  N00 -> N20;
  N01 -> N20;
  N02 -> N03;
  N03 -> N04;
  N04 -> N20;
  N04 -> N22;
  N05 -> N06;
  N06 -> N07;
  N07 -> N20;
  N07 -> N23;
  N08 -> N09;
  N09 -> N10;
  N10 -> N20;
  N10 -> N24;
  N11 -> N12;
  N12 -> N13;
  N13 -> N20;
  N13 -> N25;
  N14 -> N15;
  N15 -> N16;
  N16 -> N20;
  N16 -> N26;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N19 -> N27;
  N20 -> N21;
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
  N04[label=2];
  N05[label=1];
  N06[label=ToReal];
  N07[label="Bernoulli(logits)"];
  N08[label=Sample];
  N09[label=ToReal];
  N10[label="Bernoulli(logits)"];
  N11[label=Sample];
  N12[label=ToMatrix];
  N13[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N06;
  N03 -> N09;
  N04 -> N12;
  N05 -> N12;
  N06 -> N07;
  N07 -> N08;
  N08 -> N12;
  N09 -> N10;
  N10 -> N11;
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
  N00[label=3];
  N01[label=2];
  N02[label=1.0];
  N03[label=HalfCauchy];
  N04[label=Sample];
  N05[label=0.25];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label=ToReal];
  N09[label=2.0];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=StudentT];
  N13[label=Sample];
  N14[label=HalfCauchy];
  N15[label=Sample];
  N16[label=0.75];
  N17[label=Bernoulli];
  N18[label=Sample];
  N19[label=ToReal];
  N20[label=3.0];
  N21[label=Normal];
  N22[label=Sample];
  N23[label=StudentT];
  N24[label=Sample];
  N25[label=HalfCauchy];
  N26[label=Sample];
  N27[label=0.5];
  N28[label=Bernoulli];
  N29[label=Sample];
  N30[label=ToReal];
  N31[label=4.0];
  N32[label=Normal];
  N33[label=Sample];
  N34[label=StudentT];
  N35[label=Sample];
  N36[label=0.125];
  N37[label=Bernoulli];
  N38[label=Sample];
  N39[label=ToReal];
  N40[label=Normal];
  N41[label=Sample];
  N42[label=StudentT];
  N43[label=Sample];
  N44[label=0.875];
  N45[label=Bernoulli];
  N46[label=Sample];
  N47[label=ToReal];
  N48[label=Normal];
  N49[label=Sample];
  N50[label=StudentT];
  N51[label=Sample];
  N52[label=0.625];
  N53[label=Bernoulli];
  N54[label=Sample];
  N55[label=ToReal];
  N56[label=Normal];
  N57[label=Sample];
  N58[label=StudentT];
  N59[label=Sample];
  N60[label=ToMatrix];
  N61[label=Query];
  N00 -> N60;
  N01 -> N60;
  N02 -> N03;
  N03 -> N04;
  N04 -> N12;
  N04 -> N12;
  N04 -> N42;
  N04 -> N42;
  N05 -> N06;
  N06 -> N07;
  N07 -> N08;
  N08 -> N10;
  N09 -> N10;
  N09 -> N14;
  N09 -> N40;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N60;
  N14 -> N15;
  N15 -> N23;
  N15 -> N23;
  N15 -> N50;
  N15 -> N50;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N19 -> N21;
  N20 -> N21;
  N20 -> N25;
  N20 -> N48;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N60;
  N25 -> N26;
  N26 -> N34;
  N26 -> N34;
  N26 -> N58;
  N26 -> N58;
  N27 -> N28;
  N28 -> N29;
  N29 -> N30;
  N30 -> N32;
  N31 -> N32;
  N31 -> N56;
  N32 -> N33;
  N33 -> N34;
  N34 -> N35;
  N35 -> N60;
  N36 -> N37;
  N37 -> N38;
  N38 -> N39;
  N39 -> N40;
  N40 -> N41;
  N41 -> N42;
  N42 -> N43;
  N43 -> N60;
  N44 -> N45;
  N45 -> N46;
  N46 -> N47;
  N47 -> N48;
  N48 -> N49;
  N49 -> N50;
  N50 -> N51;
  N51 -> N60;
  N52 -> N53;
  N53 -> N54;
  N54 -> N55;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N58 -> N59;
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
  N00[label=2];
  N01[label=1];
  N02[label=2.0];
  N03[label=3.0];
  N04[label=Beta];
  N05[label=Sample];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label=4.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=Bernoulli];
  N12[label=Sample];
  N13[label=ToMatrix];
  N14[label=Query];
  N15[label=Flat];
  N16[label=Sample];
  N17[label=Bernoulli];
  N18[label=Sample];
  N19[label=Sample];
  N20[label=Bernoulli];
  N21[label=Sample];
  N22[label=ToMatrix];
  N23[label=Query];
  N00 -> N13;
  N00 -> N22;
  N01 -> N13;
  N01 -> N22;
  N02 -> N04;
  N02 -> N09;
  N03 -> N04;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N07 -> N13;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N15 -> N16;
  N15 -> N19;
  N16 -> N17;
  N17 -> N18;
  N18 -> N22;
  N19 -> N20;
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
  N00[label=2];
  N01[label=2.0];
  N02[label=3.0];
  N03[label=Beta];
  N04[label=Sample];
  N05[label=ToPosReal];
  N06[label=5.0];
  N07[label="+"];
  N08[label=4.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=6.0];
  N13[label="+"];
  N14[label=7.0];
  N15[label="+"];
  N16[label=8.0];
  N17[label="+"];
  N18[label=ToMatrix];
  N19[label=Query];
  N00 -> N18;
  N00 -> N18;
  N01 -> N03;
  N01 -> N09;
  N02 -> N03;
  N03 -> N04;
  N04 -> N05;
  N05 -> N07;
  N05 -> N15;
  N06 -> N07;
  N07 -> N18;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N11 -> N17;
  N12 -> N13;
  N13 -> N18;
  N14 -> N15;
  N15 -> N18;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
}

"""
        self.assertEqual(expected.strip(), observed.strip())
