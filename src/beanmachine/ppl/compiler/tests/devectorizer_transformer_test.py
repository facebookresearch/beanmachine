# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.copy_and_replace import copy_and_replace
from beanmachine.ppl.compiler.devectorizer_transformer import Devectorizer
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch import mm, tensor
from torch.distributions import Bernoulli, Gamma, HalfCauchy, Normal, StudentT


@bm.random_variable
def norm_tensor(n):
    return Normal(tensor([0.0, 0.5]), tensor([0.6, 1.0]))


class DevectorizeTransformerTest(unittest.TestCase):
    def test_needs_transform_because_parent_cannot_be_merged(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([norm_tensor(0)], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Devectorizer(c, s)
        )
        observed = to_dot(transformed_graph)
        expected = """
digraph "graph" {
  N00[label="[0.0,0.5]"];
  N01[label=0];
  N02[label=index];
  N03[label="[0.6000000238418579,1.0]"];
  N04[label=index];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=index];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=Tensor];
  N13[label=Query];
  N00 -> N02[label=left];
  N00 -> N08[label=left];
  N01 -> N02[label=right];
  N01 -> N04[label=right];
  N02 -> N05[label=mu];
  N03 -> N04[label=left];
  N03 -> N09[label=left];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N12[label=left];
  N07 -> N08[label=right];
  N07 -> N09[label=right];
  N08 -> N10[label=mu];
  N09 -> N10[label=sigma];
  N10 -> N11[label=operand];
  N11 -> N12[label=right];
  N12 -> N13[label=operator];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_transform_multiple_operands(self) -> None:
        _y_obs = tensor([33.3, 68.3])

        @bm.random_variable
        def sigma_out():
            return Gamma(1, 1)

        @bm.functional
        def multiple_operands():
            mu = norm_tensor(0)
            ns = Normal(mu, sigma_out())
            return ns.log_prob(_y_obs)

        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([multiple_operands()], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Devectorizer(c, s)
        )
        observed = to_dot(transformed_graph)
        expected = """
digraph "graph" {
  N00[label="[0.0,0.5]"];
  N01[label=0];
  N02[label=index];
  N03[label="[0.6000000238418579,1.0]"];
  N04[label=index];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1.0];
  N08[label=Gamma];
  N09[label=Sample];
  N10[label=Normal];
  N11[label="[33.29999923706055,68.30000305175781]"];
  N12[label=index];
  N13[label=LogProb];
  N14[label=1];
  N15[label=index];
  N16[label=index];
  N17[label=Normal];
  N18[label=Sample];
  N19[label=Normal];
  N20[label=index];
  N21[label=LogProb];
  N22[label=Tensor];
  N23[label=Query];
  N00 -> N02[label=left];
  N00 -> N15[label=left];
  N01 -> N02[label=right];
  N01 -> N04[label=right];
  N01 -> N12[label=right];
  N02 -> N05[label=mu];
  N03 -> N04[label=left];
  N03 -> N16[label=left];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N10[label=mu];
  N07 -> N08[label=concentration];
  N07 -> N08[label=rate];
  N08 -> N09[label=operand];
  N09 -> N10[label=sigma];
  N09 -> N19[label=sigma];
  N10 -> N13[label=distribution];
  N11 -> N12[label=left];
  N11 -> N20[label=left];
  N12 -> N13[label=value];
  N13 -> N22[label=left];
  N14 -> N15[label=right];
  N14 -> N16[label=right];
  N14 -> N20[label=right];
  N15 -> N17[label=mu];
  N16 -> N17[label=sigma];
  N17 -> N18[label=operand];
  N18 -> N19[label=mu];
  N19 -> N21[label=distribution];
  N20 -> N21[label=value];
  N21 -> N22[label=right];
  N22 -> N23[label=operator];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_needs_merge(self) -> None:
        @bm.functional
        def foo():
            return mm(tensor([2.0, 7.5]), norm_tensor(0))

        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([foo()], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Devectorizer(c, s)
        )
        observed = to_dot(transformed_graph)
        expected = """
        digraph "graph" {
  N00[label="[2.0,7.5]"];
  N01[label="[0.0,0.5]"];
  N02[label=0];
  N03[label=index];
  N04[label="[0.6000000238418579,1.0]"];
  N05[label=index];
  N06[label=Normal];
  N07[label=Sample];
  N08[label=1];
  N09[label=index];
  N10[label=index];
  N11[label=Normal];
  N12[label=Sample];
  N13[label=Tensor];
  N14[label="@"];
  N15[label=Query];
  N00 -> N14[label=left];
  N01 -> N03[label=left];
  N01 -> N09[label=left];
  N02 -> N03[label=right];
  N02 -> N05[label=right];
  N03 -> N06[label=mu];
  N04 -> N05[label=left];
  N04 -> N10[label=left];
  N05 -> N06[label=sigma];
  N06 -> N07[label=operand];
  N07 -> N13[label=left];
  N08 -> N09[label=right];
  N08 -> N10[label=right];
  N09 -> N11[label=mu];
  N10 -> N11[label=sigma];
  N11 -> N12[label=operand];
  N12 -> N13[label=right];
  N13 -> N14[label=right];
  N14 -> N15[label=operator];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_broadcast(self) -> None:
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

        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([studentt_2_3()], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Devectorizer(c, s)
        )
        observed = to_dot(transformed_graph)
        expected = """
        digraph "graph" {
  N00[label="[1.0,2.0,3.0]"];
  N01[label=0];
  N02[label=index];
  N03[label=HalfCauchy];
  N04[label=Sample];
  N05[label="[[0.25,0.75,0.5],\\\\n[0.125,0.875,0.625]]"];
  N06[label=index];
  N07[label=index];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label="[2.0,3.0,4.0]"];
  N11[label=index];
  N12[label=Normal];
  N13[label=Sample];
  N14[label=StudentT];
  N15[label=Sample];
  N16[label=1];
  N17[label=index];
  N18[label=HalfCauchy];
  N19[label=Sample];
  N20[label=index];
  N21[label=Bernoulli];
  N22[label=Sample];
  N23[label=index];
  N24[label=Normal];
  N25[label=Sample];
  N26[label=StudentT];
  N27[label=Sample];
  N28[label=2];
  N29[label=index];
  N30[label=HalfCauchy];
  N31[label=Sample];
  N32[label=index];
  N33[label=Bernoulli];
  N34[label=Sample];
  N35[label=index];
  N36[label=Normal];
  N37[label=Sample];
  N38[label=StudentT];
  N39[label=Sample];
  N40[label=index];
  N41[label=index];
  N42[label=Bernoulli];
  N43[label=Sample];
  N44[label=Normal];
  N45[label=Sample];
  N46[label=StudentT];
  N47[label=Sample];
  N48[label=index];
  N49[label=Bernoulli];
  N50[label=Sample];
  N51[label=Normal];
  N52[label=Sample];
  N53[label=StudentT];
  N54[label=Sample];
  N55[label=index];
  N56[label=Bernoulli];
  N57[label=Sample];
  N58[label=Normal];
  N59[label=Sample];
  N60[label=StudentT];
  N61[label=Sample];
  N62[label=Tensor];
  N63[label=Query];
  N00 -> N02[label=left];
  N00 -> N17[label=left];
  N00 -> N29[label=left];
  N01 -> N02[label=right];
  N01 -> N06[label=right];
  N01 -> N07[label=right];
  N01 -> N11[label=right];
  N01 -> N41[label=right];
  N02 -> N03[label=scale];
  N03 -> N04[label=operand];
  N04 -> N14[label=df];
  N04 -> N14[label=scale];
  N04 -> N46[label=df];
  N04 -> N46[label=scale];
  N05 -> N06[label=left];
  N05 -> N40[label=left];
  N06 -> N07[label=left];
  N06 -> N20[label=left];
  N06 -> N32[label=left];
  N07 -> N08[label=probability];
  N08 -> N09[label=operand];
  N09 -> N12[label=mu];
  N10 -> N11[label=left];
  N10 -> N23[label=left];
  N10 -> N35[label=left];
  N11 -> N12[label=sigma];
  N11 -> N44[label=sigma];
  N12 -> N13[label=operand];
  N13 -> N14[label=loc];
  N14 -> N15[label=operand];
  N15 -> N62[label=0];
  N16 -> N17[label=right];
  N16 -> N20[label=right];
  N16 -> N23[label=right];
  N16 -> N40[label=right];
  N16 -> N48[label=right];
  N17 -> N18[label=scale];
  N18 -> N19[label=operand];
  N19 -> N26[label=df];
  N19 -> N26[label=scale];
  N19 -> N53[label=df];
  N19 -> N53[label=scale];
  N20 -> N21[label=probability];
  N21 -> N22[label=operand];
  N22 -> N24[label=mu];
  N23 -> N24[label=sigma];
  N23 -> N51[label=sigma];
  N24 -> N25[label=operand];
  N25 -> N26[label=loc];
  N26 -> N27[label=operand];
  N27 -> N62[label=1];
  N28 -> N29[label=right];
  N28 -> N32[label=right];
  N28 -> N35[label=right];
  N28 -> N55[label=right];
  N29 -> N30[label=scale];
  N30 -> N31[label=operand];
  N31 -> N38[label=df];
  N31 -> N38[label=scale];
  N31 -> N60[label=df];
  N31 -> N60[label=scale];
  N32 -> N33[label=probability];
  N33 -> N34[label=operand];
  N34 -> N36[label=mu];
  N35 -> N36[label=sigma];
  N35 -> N58[label=sigma];
  N36 -> N37[label=operand];
  N37 -> N38[label=loc];
  N38 -> N39[label=operand];
  N39 -> N62[label=2];
  N40 -> N41[label=left];
  N40 -> N48[label=left];
  N40 -> N55[label=left];
  N41 -> N42[label=probability];
  N42 -> N43[label=operand];
  N43 -> N44[label=mu];
  N44 -> N45[label=operand];
  N45 -> N46[label=loc];
  N46 -> N47[label=operand];
  N47 -> N62[label=3];
  N48 -> N49[label=probability];
  N49 -> N50[label=operand];
  N50 -> N51[label=mu];
  N51 -> N52[label=operand];
  N52 -> N53[label=loc];
  N53 -> N54[label=operand];
  N54 -> N62[label=4];
  N55 -> N56[label=probability];
  N56 -> N57[label=operand];
  N57 -> N58[label=mu];
  N58 -> N59[label=operand];
  N59 -> N60[label=loc];
  N60 -> N61[label=operand];
  N61 -> N62[label=5];
  N62 -> N63[label=operator];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_failure(self) -> None:
        # in order to devectorize correctly, all sizes must be known.
        # note that "ns.log_prob" has an unsizable node since we are asking
        # what the log prob of a tensor of size 3 is with respect to a distribution
        # whose samples are of size 2.
        _y_obs = tensor([33.3, 68.3, 6.7])

        @bm.random_variable
        def sigma_out():
            return Gamma(1, 1)

        @bm.functional
        def unsizable():
            mu = norm_tensor(0)
            ns = Normal(mu, sigma_out())
            return ns.log_prob(_y_obs)

        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([unsizable()], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Devectorizer(c, s)
        )
        if len(error_report.errors) == 1:
            error = error_report.errors[0].__str__()
            expected = """
The node log_prob cannot be sized.The operand sizes may be incompatible or the size may not be computable at compile time. The operand sizes are: [torch.Size([2]), torch.Size([3])]
The unsizable node was created in function call unsizable().
            """
            self.assertEqual(expected.strip(), error.strip())
        else:
            self.fail(
                "A single error message should have been generated since the sizer cannot size every node"
            )
