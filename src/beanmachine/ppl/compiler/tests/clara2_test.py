#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.hint import log1mexp
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


"""
In this version of the CLARA model we change the generative
process of each labeler's confusion matrix. For each row in
the labeler's confusion matrix:

diagonal_element ~ 1 - 0.5 * Beta(a, b)
other_elements   ~ (1 - diagonal_element) * Dirichlet(K-1)

This way, diagonal_element + other_elements = 1
but we have forced diagonal_element > 0.5.
"""


NUM_LABELS = 3
BRONZE = 0
SILVER = 1
GOLD = 2

NUM_LABELERS = 3
SUE = 0
BOB = 1
EVE = 2

NUM_ITEMS = 4

# The labels given to items by labelers; a ragged array with
# NUM_ITEMS rows. Each row has no more than NUM_LABELERS labels.
ITEM_LABELS = [
    [GOLD, GOLD, SILVER],
    [BRONZE, SILVER],
    [SILVER, SILVER],
    [BRONZE, BRONZE],
]

# The labelers who labeled the items; must have exact same shape as ITEM_LABELS.
IDX_LABELERS = [
    [SUE, BOB, EVE],
    [SUE, EVE],
    [SUE, BOB],
    [BOB, EVE],
]

# The expert confusion matrix is a NUM_LABELS x NUM_LABELS matrix giving the
# probability of an expert producing each possible label for an item with a given
# true label. The first row for instance is "if an expert is given an item that
# should be labeled BRONZE, there is a 90% chance that the item is labeled BRONZE,
# 7% SILVER, 3% GOLD"
EXPERT_CONF_MATRIX = tensor(
    [
        [0.9, 0.7, 0.3],
        [0.5, 0.9, 0.5],
        [0.3, 0.7, 0.9],
    ]
)


# The true label of each item, or -1 if the true label is unknown.
TRUE_LABELS = [GOLD, SILVER, -1, BRONZE]


# Produces a simplex of length NUM_LABELS.
# Each entry is the probability that it is the true label of random item.
@bm.random_variable
def prevalence():
    PREVALENCE_PRIOR = torch.ones(NUM_LABELS)
    return dist.Dirichlet(PREVALENCE_PRIOR)


# Used to compute the probability that the labeler *correctly* labels an
# item with the given true label. In the next method we force this
# quantity to be > 0.5.
@bm.random_variable
def confusion_diag(labeler, true_label):
    SOME_CONSTANT1 = 2.0
    SOME_CONSTANT2 = 2.0
    return dist.Beta(SOME_CONSTANT1, SOME_CONSTANT2)


# Force the probability of a correct label to be >0.5 and
# take its log. Note that this is stable because the operand
# of the log is not close to zero.
@bm.functional
def log_constrained_confusion_diag(labeler, true_label):
    return torch.log(1 - 0.5 * confusion_diag(labeler, true_label))


# This is the log-probability of getting *any* incorrect label; it is
# the inverse of the probability of a correct label. Again, this computation
# is stable even if the probability of getting an incorrect label is
# close to zero.
@bm.functional
def log1m_constrained_confusion_diag(labeler, true_label):
    # TODO: Fix the compiler so that we recognize log(1-exp(x)) and replace
    # it automatically with a LOG1MEXP node rather than relying on a call
    # to a hint helper.
    return log1mexp(log_constrained_confusion_diag(labeler, true_label))


# Produces a simplex of length NUM_LABELS-1.
# This is used to compute probability that the labeler *incorrectly* labels an
# item with the given true label, so there are NUM_LABELS-1 possibilities.
@bm.random_variable
def confusion_non_diag(labeler, true_label):
    return dist.Dirichlet(torch.ones(NUM_LABELS - 1))


# Produces a NUM_LABELS x NUM_LABELS matrix which answers the question
# "For an item with a given true label, what is the probability that
# this labeler will assign each possible label?"
@bm.functional
def log_confusion_matrix(labeler):
    # Original code was
    #
    # log_conf_matrix = torch.ones(NUM_LABELS, NUM_LABELS)
    #
    # but we do not support mutation of a tensor with stochastic graph
    # elements in BMG.  Instead, construct a list, mutate that, and
    # then turn it into a tensor.

    log_conf_matrix = [[None] * NUM_LABELS for _ in range(NUM_LABELS)]
    for true_label in range(NUM_LABELS):
        # Start by filling in the diagonal; the diagonal is the case where the
        # labeler gets it right.
        log_conf_matrix[true_label][true_label] = log_constrained_confusion_diag(
            labeler, true_label
        )
        # Now fill in all the cases where the labeler gets it wrong.
        for observed_label in range(true_label):
            log_conf_matrix[true_label][
                observed_label
            ] = log1m_constrained_confusion_diag(labeler, true_label) + torch.log(
                confusion_non_diag(labeler, true_label)[observed_label]
            )
        for observed_label in range(true_label + 1, NUM_LABELS):
            log_conf_matrix[true_label][
                observed_label
            ] = log1m_constrained_confusion_diag(labeler, true_label) + torch.log(
                confusion_non_diag(labeler, true_label)[observed_label - 1]
            )
    return tensor(log_conf_matrix)


@bm.functional
def log_item_prob(item, true_label):
    prob = torch.log(prevalence()[true_label])
    for label_index in range(len(ITEM_LABELS[item])):
        label = ITEM_LABELS[item][label_index]
        labeler = IDX_LABELERS[item][label_index]
        # TODO: The more natural way to write this would be [true_label, label]
        # but we do not support stochastic tuple indices yet.
        # TODO: The compiler still does not handle models that contain +=.
        prob = prob + log_confusion_matrix(labeler)[true_label][label]
    if TRUE_LABELS[item] != -1:
        # TODO: Similarly with the indexing here.
        prob = prob + torch.log(EXPERT_CONF_MATRIX[true_label][TRUE_LABELS[item]])
    return prob


# log of joint prob of labels, prevalence, confusion matrix
@bm.random_variable
def target(item):
    all_item_probs = [
        log_item_prob(item, true_label) for true_label in range(NUM_LABELS)
    ]
    joint_log_prob = torch.logsumexp(tensor(all_item_probs), dim=0)
    return dist.Bernoulli(torch.exp(joint_log_prob))


observations = {target(item): tensor(1.0) for item in range(NUM_ITEMS)}
# Given observations of labeler's choice of label for each item we wish to know:
# * for every item, what is the probability of each possible label being correct?
# * what is the true prevalence of each label?
# * what are the confusion matrices for each labeler?
queries = (
    [
        log_item_prob(item, true_label)
        for item in range(NUM_ITEMS)
        for true_label in range(NUM_LABELS)
    ]
    + [prevalence()]
    + [log_confusion_matrix(labeler) for labeler in range(NUM_LABELERS)]
)


class Clara2Test(unittest.TestCase):
    def test_clara2_inference(self) -> None:
        self.maxDiff = None

        results = BMGInference().infer(queries, observations, 100)
        item_0_bronze = results[log_item_prob(0, BRONZE)].mean()
        item_0_silver = results[log_item_prob(0, SILVER)].mean()
        item_0_gold = results[log_item_prob(0, GOLD)].mean()
        cm_sue = results[log_confusion_matrix(0)].exp().mean(dim=1)

        # These are non-normalized log-probabilities that item 0 is of each
        # possible label; the softmax function would normalize them.  Note
        # that the item is judged to be much more likely to be gold than silver,
        # and much more likely to be silver than bronze.
        self.assertAlmostEqual(first=-9.75, second=item_0_bronze, delta=1.0)
        self.assertAlmostEqual(first=-7.48, second=item_0_silver, delta=1.0)
        self.assertAlmostEqual(first=-4.13, second=item_0_gold, delta=1.0)

        # The confusion matrix gives the probability that labeler Sue assigns
        # a given label to an item. For example, we infer that when given an item
        # whose true label is bronze, Sue assigns bronze 76% of the time,
        # silver 11% of the time and gold 13% of the time.

        self.assertAlmostEqual(first=0.76, second=cm_sue[0, BRONZE, BRONZE], delta=0.1)
        self.assertAlmostEqual(first=0.11, second=cm_sue[0, BRONZE, SILVER], delta=0.1)
        self.assertAlmostEqual(first=0.13, second=cm_sue[0, BRONZE, GOLD], delta=0.1)

    def test_clara2_to_graph(self) -> None:
        self.maxDiff = None

        bmg = BMGInference()
        bmg._fix_observe_true = True

        # to_graph produces a BMG graph object and a map from your queries to
        # the query ids; these are the indices into the returned samples list
        # when you call infer on the graph, so it is handy to have that information
        # available.
        g, q = bmg.to_graph(queries, observations)
        expected = """
digraph "graph" {
  N0[label="matrix"];
  N1[label="Dirichlet"];
  N2[label="~"];
  N3[label="2"];
  N4[label="Beta"];
  N5[label="~"];
  N6[label="matrix"];
  N7[label="Dirichlet"];
  N8[label="~"];
  N9[label="~"];
  N10[label="~"];
  N11[label="~"];
  N12[label="~"];
  N13[label="~"];
  N14[label="~"];
  N15[label="~"];
  N16[label="~"];
  N17[label="~"];
  N18[label="~"];
  N19[label="~"];
  N20[label="~"];
  N21[label="~"];
  N22[label="~"];
  N23[label="~"];
  N24[label="~"];
  N25[label="0"];
  N26[label="Index"];
  N27[label="Log"];
  N28[label="3"];
  N29[label="0.5"];
  N30[label="*"];
  N31[label="Complement"];
  N32[label="Log"];
  N33[label="Log1mExp"];
  N34[label="Index"];
  N35[label="Log"];
  N36[label="+"];
  N37[label="1"];
  N38[label="Index"];
  N39[label="Log"];
  N40[label="+"];
  N41[label="*"];
  N42[label="Complement"];
  N43[label="Log"];
  N44[label="Log1mExp"];
  N45[label="Index"];
  N46[label="Log"];
  N47[label="+"];
  N48[label="Index"];
  N49[label="Log"];
  N50[label="+"];
  N51[label="*"];
  N52[label="Complement"];
  N53[label="Log"];
  N54[label="Log1mExp"];
  N55[label="Index"];
  N56[label="Log"];
  N57[label="+"];
  N58[label="Index"];
  N59[label="Log"];
  N60[label="+"];
  N61[label="ToMatrix"];
  N62[label="ColumnIndex"];
  N63[label="2"];
  N64[label="Index"];
  N65[label="*"];
  N66[label="Complement"];
  N67[label="Log"];
  N68[label="Log1mExp"];
  N69[label="Index"];
  N70[label="Log"];
  N71[label="+"];
  N72[label="Index"];
  N73[label="Log"];
  N74[label="+"];
  N75[label="*"];
  N76[label="Complement"];
  N77[label="Log"];
  N78[label="Log1mExp"];
  N79[label="Index"];
  N80[label="Log"];
  N81[label="+"];
  N82[label="Index"];
  N83[label="Log"];
  N84[label="+"];
  N85[label="*"];
  N86[label="Complement"];
  N87[label="Log"];
  N88[label="Log1mExp"];
  N89[label="Index"];
  N90[label="Log"];
  N91[label="+"];
  N92[label="Index"];
  N93[label="Log"];
  N94[label="+"];
  N95[label="ToMatrix"];
  N96[label="ColumnIndex"];
  N97[label="Index"];
  N98[label="*"];
  N99[label="Complement"];
  N100[label="Log"];
  N101[label="Log1mExp"];
  N102[label="Index"];
  N103[label="Log"];
  N104[label="+"];
  N105[label="Index"];
  N106[label="Log"];
  N107[label="+"];
  N108[label="*"];
  N109[label="Complement"];
  N110[label="Log"];
  N111[label="Log1mExp"];
  N112[label="Index"];
  N113[label="Log"];
  N114[label="+"];
  N115[label="Index"];
  N116[label="Log"];
  N117[label="+"];
  N118[label="*"];
  N119[label="Complement"];
  N120[label="Log"];
  N121[label="Log1mExp"];
  N122[label="Index"];
  N123[label="Log"];
  N124[label="+"];
  N125[label="Index"];
  N126[label="Log"];
  N127[label="+"];
  N128[label="ToMatrix"];
  N129[label="ColumnIndex"];
  N130[label="Index"];
  N131[label="-1.20397"];
  N132[label="+"];
  N133[label="Index"];
  N134[label="Log"];
  N135[label="ColumnIndex"];
  N136[label="Index"];
  N137[label="ColumnIndex"];
  N138[label="Index"];
  N139[label="ColumnIndex"];
  N140[label="Index"];
  N141[label="-0.693147"];
  N142[label="+"];
  N143[label="Index"];
  N144[label="Log"];
  N145[label="ColumnIndex"];
  N146[label="Index"];
  N147[label="ColumnIndex"];
  N148[label="Index"];
  N149[label="ColumnIndex"];
  N150[label="Index"];
  N151[label="-0.105361"];
  N152[label="+"];
  N153[label="LogSumExp"];
  N154[label="exp"];
  N155[label="ToProb"];
  N156[label="Bernoulli"];
  N157[label="~"];
  N158[label="Index"];
  N159[label="-0.356675"];
  N160[label="+"];
  N161[label="Index"];
  N162[label="+"];
  N163[label="Index"];
  N164[label="+"];
  N165[label="LogSumExp"];
  N166[label="exp"];
  N167[label="ToProb"];
  N168[label="Bernoulli"];
  N169[label="~"];
  N170[label="Index"];
  N171[label="Index"];
  N172[label="+"];
  N173[label="Index"];
  N174[label="Index"];
  N175[label="+"];
  N176[label="Index"];
  N177[label="Index"];
  N178[label="+"];
  N179[label="LogSumExp"];
  N180[label="exp"];
  N181[label="ToProb"];
  N182[label="Bernoulli"];
  N183[label="~"];
  N184[label="Index"];
  N185[label="Index"];
  N186[label="+"];
  N187[label="Index"];
  N188[label="Index"];
  N189[label="+"];
  N190[label="Index"];
  N191[label="Index"];
  N192[label="+"];
  N193[label="LogSumExp"];
  N194[label="exp"];
  N195[label="ToProb"];
  N196[label="Bernoulli"];
  N197[label="~"];
  N198[label="Factor"];
  N199[label="Factor"];
  N200[label="Factor"];
  N201[label="Factor"];
  N0 -> N1;
  N1 -> N2;
  N2 -> N26;
  N2 -> N133;
  N2 -> N143;
  N3 -> N4;
  N3 -> N4;
  N4 -> N5;
  N4 -> N9;
  N4 -> N11;
  N4 -> N13;
  N4 -> N15;
  N4 -> N17;
  N4 -> N19;
  N4 -> N21;
  N4 -> N23;
  N5 -> N30;
  N6 -> N7;
  N7 -> N8;
  N7 -> N10;
  N7 -> N12;
  N7 -> N14;
  N7 -> N16;
  N7 -> N18;
  N7 -> N20;
  N7 -> N22;
  N7 -> N24;
  N8 -> N34;
  N8 -> N38;
  N9 -> N41;
  N10 -> N45;
  N10 -> N48;
  N11 -> N51;
  N12 -> N55;
  N12 -> N58;
  N13 -> N65;
  N14 -> N69;
  N14 -> N72;
  N15 -> N75;
  N16 -> N79;
  N16 -> N82;
  N17 -> N85;
  N18 -> N89;
  N18 -> N92;
  N19 -> N98;
  N20 -> N102;
  N20 -> N105;
  N21 -> N108;
  N22 -> N112;
  N22 -> N115;
  N23 -> N118;
  N24 -> N122;
  N24 -> N125;
  N25 -> N26;
  N25 -> N34;
  N25 -> N45;
  N25 -> N55;
  N25 -> N62;
  N25 -> N69;
  N25 -> N79;
  N25 -> N89;
  N25 -> N96;
  N25 -> N102;
  N25 -> N112;
  N25 -> N122;
  N25 -> N129;
  N25 -> N158;
  N25 -> N161;
  N25 -> N163;
  N25 -> N184;
  N25 -> N185;
  N25 -> N187;
  N25 -> N188;
  N25 -> N190;
  N25 -> N191;
  N26 -> N27;
  N27 -> N132;
  N27 -> N160;
  N27 -> N172;
  N27 -> N186;
  N28 -> N61;
  N28 -> N61;
  N28 -> N95;
  N28 -> N95;
  N28 -> N128;
  N28 -> N128;
  N29 -> N30;
  N29 -> N41;
  N29 -> N51;
  N29 -> N65;
  N29 -> N75;
  N29 -> N85;
  N29 -> N98;
  N29 -> N108;
  N29 -> N118;
  N30 -> N31;
  N31 -> N32;
  N32 -> N33;
  N32 -> N61;
  N33 -> N36;
  N33 -> N40;
  N34 -> N35;
  N35 -> N36;
  N36 -> N61;
  N37 -> N38;
  N37 -> N48;
  N37 -> N58;
  N37 -> N72;
  N37 -> N82;
  N37 -> N92;
  N37 -> N105;
  N37 -> N115;
  N37 -> N125;
  N37 -> N130;
  N37 -> N133;
  N37 -> N135;
  N37 -> N137;
  N37 -> N139;
  N37 -> N140;
  N37 -> N150;
  N37 -> N170;
  N37 -> N171;
  N37 -> N173;
  N37 -> N174;
  N37 -> N176;
  N37 -> N177;
  N38 -> N39;
  N39 -> N40;
  N40 -> N61;
  N41 -> N42;
  N42 -> N43;
  N43 -> N44;
  N43 -> N61;
  N44 -> N47;
  N44 -> N50;
  N45 -> N46;
  N46 -> N47;
  N47 -> N61;
  N48 -> N49;
  N49 -> N50;
  N50 -> N61;
  N51 -> N52;
  N52 -> N53;
  N53 -> N54;
  N53 -> N61;
  N54 -> N57;
  N54 -> N60;
  N55 -> N56;
  N56 -> N57;
  N57 -> N61;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N61 -> N62;
  N61 -> N135;
  N61 -> N145;
  N62 -> N64;
  N62 -> N158;
  N62 -> N170;
  N63 -> N64;
  N63 -> N97;
  N63 -> N136;
  N63 -> N138;
  N63 -> N143;
  N63 -> N145;
  N63 -> N146;
  N63 -> N147;
  N63 -> N148;
  N63 -> N149;
  N64 -> N132;
  N65 -> N66;
  N66 -> N67;
  N67 -> N68;
  N67 -> N95;
  N68 -> N71;
  N68 -> N74;
  N69 -> N70;
  N70 -> N71;
  N71 -> N95;
  N72 -> N73;
  N73 -> N74;
  N74 -> N95;
  N75 -> N76;
  N76 -> N77;
  N77 -> N78;
  N77 -> N95;
  N78 -> N81;
  N78 -> N84;
  N79 -> N80;
  N80 -> N81;
  N81 -> N95;
  N82 -> N83;
  N83 -> N84;
  N84 -> N95;
  N85 -> N86;
  N86 -> N87;
  N87 -> N88;
  N87 -> N95;
  N88 -> N91;
  N88 -> N94;
  N89 -> N90;
  N90 -> N91;
  N91 -> N95;
  N92 -> N93;
  N93 -> N94;
  N94 -> N95;
  N95 -> N96;
  N95 -> N137;
  N95 -> N147;
  N96 -> N97;
  N96 -> N171;
  N96 -> N184;
  N97 -> N132;
  N98 -> N99;
  N99 -> N100;
  N100 -> N101;
  N100 -> N128;
  N101 -> N104;
  N101 -> N107;
  N102 -> N103;
  N103 -> N104;
  N104 -> N128;
  N105 -> N106;
  N106 -> N107;
  N107 -> N128;
  N108 -> N109;
  N109 -> N110;
  N110 -> N111;
  N110 -> N128;
  N111 -> N114;
  N111 -> N117;
  N112 -> N113;
  N113 -> N114;
  N114 -> N128;
  N115 -> N116;
  N116 -> N117;
  N117 -> N128;
  N118 -> N119;
  N119 -> N120;
  N120 -> N121;
  N120 -> N128;
  N121 -> N124;
  N121 -> N127;
  N122 -> N123;
  N123 -> N124;
  N124 -> N128;
  N125 -> N126;
  N126 -> N127;
  N127 -> N128;
  N128 -> N129;
  N128 -> N139;
  N128 -> N149;
  N129 -> N130;
  N129 -> N185;
  N130 -> N132;
  N130 -> N160;
  N131 -> N132;
  N131 -> N192;
  N132 -> N153;
  N133 -> N134;
  N134 -> N142;
  N134 -> N162;
  N134 -> N175;
  N134 -> N189;
  N135 -> N136;
  N135 -> N161;
  N135 -> N173;
  N136 -> N142;
  N137 -> N138;
  N137 -> N174;
  N137 -> N187;
  N138 -> N142;
  N139 -> N140;
  N139 -> N188;
  N140 -> N142;
  N140 -> N162;
  N141 -> N142;
  N141 -> N189;
  N142 -> N153;
  N143 -> N144;
  N144 -> N152;
  N144 -> N164;
  N144 -> N178;
  N144 -> N192;
  N145 -> N146;
  N145 -> N163;
  N145 -> N176;
  N146 -> N152;
  N147 -> N148;
  N147 -> N177;
  N147 -> N190;
  N148 -> N152;
  N149 -> N150;
  N149 -> N191;
  N150 -> N152;
  N150 -> N164;
  N151 -> N152;
  N151 -> N162;
  N151 -> N186;
  N152 -> N153;
  N153 -> N154;
  N153 -> N198;
  N154 -> N155;
  N155 -> N156;
  N156 -> N157;
  N158 -> N160;
  N159 -> N160;
  N159 -> N164;
  N160 -> N165;
  N161 -> N162;
  N162 -> N165;
  N163 -> N164;
  N164 -> N165;
  N165 -> N166;
  N165 -> N199;
  N166 -> N167;
  N167 -> N168;
  N168 -> N169;
  N170 -> N172;
  N171 -> N172;
  N172 -> N179;
  N173 -> N175;
  N174 -> N175;
  N175 -> N179;
  N176 -> N178;
  N177 -> N178;
  N178 -> N179;
  N179 -> N180;
  N179 -> N200;
  N180 -> N181;
  N181 -> N182;
  N182 -> N183;
  N184 -> N186;
  N185 -> N186;
  N186 -> N193;
  N187 -> N189;
  N188 -> N189;
  N189 -> N193;
  N190 -> N192;
  N191 -> N192;
  N192 -> N193;
  N193 -> N194;
  N193 -> N201;
  N194 -> N195;
  N195 -> N196;
  N196 -> N197;
  O0[label="Observation"];
  N198 -> O0;
  O1[label="Observation"];
  N199 -> O1;
  O2[label="Observation"];
  N200 -> O2;
  O3[label="Observation"];
  N201 -> O3;
  Q0[label="Query"];
  N132 -> Q0;
  Q1[label="Query"];
  N142 -> Q1;
  Q2[label="Query"];
  N152 -> Q2;
  Q3[label="Query"];
  N160 -> Q3;
  Q4[label="Query"];
  N162 -> Q4;
  Q5[label="Query"];
  N164 -> Q5;
  Q6[label="Query"];
  N172 -> Q6;
  Q7[label="Query"];
  N175 -> Q7;
  Q8[label="Query"];
  N178 -> Q8;
  Q9[label="Query"];
  N186 -> Q9;
  Q10[label="Query"];
  N189 -> Q10;
  Q11[label="Query"];
  N192 -> Q11;
  Q12[label="Query"];
  N2 -> Q12;
  Q13[label="Query"];
  N61 -> Q13;
  Q14[label="Query"];
  N95 -> Q14;
  Q15[label="Query"];
  N128 -> Q15;
}"""
        observed = g.to_dot()
        self.assertEqual(expected.strip(), observed.strip())
        self.assertEqual(len(queries), len(q))
        # Query node Q12 is the query on prevalence()
        self.assertEqual(12, q[prevalence()])
