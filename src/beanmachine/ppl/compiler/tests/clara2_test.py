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
  N28[label="0.5"];
  N29[label="*"];
  N30[label="Complement"];
  N31[label="Log"];
  N32[label="Log1mExp"];
  N33[label="1"];
  N34[label="Index"];
  N35[label="Log"];
  N36[label="+"];
  N37[label="*"];
  N38[label="Complement"];
  N39[label="Log"];
  N40[label="Log1mExp"];
  N41[label="Index"];
  N42[label="Log"];
  N43[label="+"];
  N44[label="*"];
  N45[label="Complement"];
  N46[label="Log"];
  N47[label="Log1mExp"];
  N48[label="Index"];
  N49[label="Log"];
  N50[label="+"];
  N51[label="-1.20397"];
  N52[label="+"];
  N53[label="Index"];
  N54[label="Log"];
  N55[label="*"];
  N56[label="Complement"];
  N57[label="Log"];
  N58[label="Log1mExp"];
  N59[label="Index"];
  N60[label="Log"];
  N61[label="+"];
  N62[label="*"];
  N63[label="Complement"];
  N64[label="Log"];
  N65[label="Log1mExp"];
  N66[label="Index"];
  N67[label="Log"];
  N68[label="+"];
  N69[label="*"];
  N70[label="Complement"];
  N71[label="Log"];
  N72[label="-0.693147"];
  N73[label="+"];
  N74[label="2"];
  N75[label="Index"];
  N76[label="Log"];
  N77[label="*"];
  N78[label="Complement"];
  N79[label="Log"];
  N80[label="*"];
  N81[label="Complement"];
  N82[label="Log"];
  N83[label="*"];
  N84[label="Complement"];
  N85[label="Log"];
  N86[label="Log1mExp"];
  N87[label="Index"];
  N88[label="Log"];
  N89[label="+"];
  N90[label="-0.105361"];
  N91[label="+"];
  N92[label="LogSumExp"];
  N93[label="exp"];
  N94[label="ToProb"];
  N95[label="Bernoulli"];
  N96[label="~"];
  N97[label="-0.356675"];
  N98[label="+"];
  N99[label="Index"];
  N100[label="Log"];
  N101[label="+"];
  N102[label="+"];
  N103[label="Log1mExp"];
  N104[label="Index"];
  N105[label="Log"];
  N106[label="+"];
  N107[label="+"];
  N108[label="LogSumExp"];
  N109[label="exp"];
  N110[label="ToProb"];
  N111[label="Bernoulli"];
  N112[label="~"];
  N113[label="Index"];
  N114[label="Log"];
  N115[label="+"];
  N116[label="Index"];
  N117[label="Log"];
  N118[label="+"];
  N119[label="+"];
  N120[label="+"];
  N121[label="Index"];
  N122[label="Log"];
  N123[label="+"];
  N124[label="Log1mExp"];
  N125[label="Index"];
  N126[label="Log"];
  N127[label="+"];
  N128[label="+"];
  N129[label="LogSumExp"];
  N130[label="exp"];
  N131[label="ToProb"];
  N132[label="Bernoulli"];
  N133[label="~"];
  N134[label="+"];
  N135[label="Index"];
  N136[label="Log"];
  N137[label="+"];
  N138[label="Log1mExp"];
  N139[label="Index"];
  N140[label="Log"];
  N141[label="+"];
  N142[label="+"];
  N143[label="Index"];
  N144[label="Log"];
  N145[label="+"];
  N146[label="Index"];
  N147[label="Log"];
  N148[label="+"];
  N149[label="+"];
  N150[label="LogSumExp"];
  N151[label="exp"];
  N152[label="ToProb"];
  N153[label="Bernoulli"];
  N154[label="~"];
  N155[label="3"];
  N156[label="ToMatrix"];
  N157[label="ToMatrix"];
  N158[label="Index"];
  N159[label="Log"];
  N160[label="+"];
  N161[label="Index"];
  N162[label="Log"];
  N163[label="+"];
  N164[label="ToMatrix"];
  N165[label="Factor"];
  N166[label="Factor"];
  N167[label="Factor"];
  N168[label="Factor"];
  N0 -> N1;
  N1 -> N2;
  N2 -> N26;
  N2 -> N53;
  N2 -> N75;
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
  N5 -> N29;
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
  N8 -> N113;
  N9 -> N55;
  N10 -> N59;
  N10 -> N99;
  N11 -> N77;
  N12 -> N104;
  N12 -> N121;
  N13 -> N37;
  N14 -> N41;
  N14 -> N116;
  N15 -> N62;
  N16 -> N66;
  N16 -> N135;
  N17 -> N80;
  N18 -> N125;
  N18 -> N143;
  N19 -> N44;
  N20 -> N48;
  N20 -> N158;
  N21 -> N69;
  N22 -> N139;
  N22 -> N161;
  N23 -> N83;
  N24 -> N87;
  N24 -> N146;
  N25 -> N26;
  N25 -> N48;
  N25 -> N99;
  N25 -> N104;
  N25 -> N113;
  N25 -> N116;
  N25 -> N135;
  N25 -> N139;
  N25 -> N143;
  N25 -> N146;
  N26 -> N27;
  N27 -> N52;
  N27 -> N98;
  N27 -> N119;
  N27 -> N134;
  N28 -> N29;
  N28 -> N37;
  N28 -> N44;
  N28 -> N55;
  N28 -> N62;
  N28 -> N69;
  N28 -> N77;
  N28 -> N80;
  N28 -> N83;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
  N31 -> N98;
  N31 -> N156;
  N32 -> N36;
  N32 -> N115;
  N33 -> N34;
  N33 -> N41;
  N33 -> N53;
  N33 -> N59;
  N33 -> N66;
  N33 -> N87;
  N33 -> N121;
  N33 -> N125;
  N33 -> N158;
  N33 -> N161;
  N34 -> N35;
  N35 -> N36;
  N36 -> N52;
  N36 -> N156;
  N37 -> N38;
  N38 -> N39;
  N39 -> N40;
  N39 -> N134;
  N39 -> N157;
  N40 -> N43;
  N40 -> N118;
  N41 -> N42;
  N42 -> N43;
  N43 -> N52;
  N43 -> N157;
  N44 -> N45;
  N45 -> N46;
  N46 -> N47;
  N46 -> N134;
  N46 -> N164;
  N47 -> N50;
  N47 -> N160;
  N48 -> N49;
  N49 -> N50;
  N50 -> N52;
  N50 -> N98;
  N50 -> N164;
  N51 -> N52;
  N51 -> N149;
  N52 -> N92;
  N53 -> N54;
  N54 -> N73;
  N54 -> N102;
  N54 -> N120;
  N54 -> N142;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N57 -> N120;
  N57 -> N156;
  N58 -> N61;
  N58 -> N101;
  N59 -> N60;
  N60 -> N61;
  N61 -> N73;
  N61 -> N156;
  N62 -> N63;
  N63 -> N64;
  N64 -> N65;
  N64 -> N120;
  N64 -> N157;
  N65 -> N68;
  N65 -> N137;
  N66 -> N67;
  N67 -> N68;
  N68 -> N73;
  N68 -> N157;
  N69 -> N70;
  N70 -> N71;
  N71 -> N73;
  N71 -> N102;
  N71 -> N138;
  N71 -> N164;
  N72 -> N73;
  N72 -> N142;
  N73 -> N92;
  N74 -> N75;
  N75 -> N76;
  N76 -> N91;
  N76 -> N107;
  N76 -> N128;
  N76 -> N149;
  N77 -> N78;
  N78 -> N79;
  N79 -> N91;
  N79 -> N103;
  N79 -> N156;
  N80 -> N81;
  N81 -> N82;
  N82 -> N91;
  N82 -> N124;
  N82 -> N157;
  N83 -> N84;
  N84 -> N85;
  N85 -> N86;
  N85 -> N164;
  N86 -> N89;
  N86 -> N148;
  N87 -> N88;
  N88 -> N89;
  N89 -> N91;
  N89 -> N107;
  N89 -> N164;
  N90 -> N91;
  N90 -> N102;
  N90 -> N134;
  N91 -> N92;
  N92 -> N93;
  N92 -> N165;
  N93 -> N94;
  N94 -> N95;
  N95 -> N96;
  N97 -> N98;
  N97 -> N107;
  N98 -> N108;
  N99 -> N100;
  N100 -> N101;
  N101 -> N102;
  N101 -> N156;
  N102 -> N108;
  N103 -> N106;
  N103 -> N123;
  N104 -> N105;
  N105 -> N106;
  N106 -> N107;
  N106 -> N156;
  N107 -> N108;
  N108 -> N109;
  N108 -> N166;
  N109 -> N110;
  N110 -> N111;
  N111 -> N112;
  N113 -> N114;
  N114 -> N115;
  N115 -> N119;
  N115 -> N156;
  N116 -> N117;
  N117 -> N118;
  N118 -> N119;
  N118 -> N157;
  N119 -> N129;
  N120 -> N129;
  N121 -> N122;
  N122 -> N123;
  N123 -> N128;
  N123 -> N156;
  N124 -> N127;
  N124 -> N145;
  N125 -> N126;
  N126 -> N127;
  N127 -> N128;
  N127 -> N157;
  N128 -> N129;
  N129 -> N130;
  N129 -> N167;
  N130 -> N131;
  N131 -> N132;
  N132 -> N133;
  N134 -> N150;
  N135 -> N136;
  N136 -> N137;
  N137 -> N142;
  N137 -> N157;
  N138 -> N141;
  N138 -> N163;
  N139 -> N140;
  N140 -> N141;
  N141 -> N142;
  N141 -> N164;
  N142 -> N150;
  N143 -> N144;
  N144 -> N145;
  N145 -> N149;
  N145 -> N157;
  N146 -> N147;
  N147 -> N148;
  N148 -> N149;
  N148 -> N164;
  N149 -> N150;
  N150 -> N151;
  N150 -> N168;
  N151 -> N152;
  N152 -> N153;
  N153 -> N154;
  N155 -> N156;
  N155 -> N156;
  N155 -> N157;
  N155 -> N157;
  N155 -> N164;
  N155 -> N164;
  N158 -> N159;
  N159 -> N160;
  N160 -> N164;
  N161 -> N162;
  N162 -> N163;
  N163 -> N164;
  O0[label="Observation"];
  N165 -> O0;
  O1[label="Observation"];
  N166 -> O1;
  O2[label="Observation"];
  N167 -> O2;
  O3[label="Observation"];
  N168 -> O3;
  Q0[label="Query"];
  N52 -> Q0;
  Q1[label="Query"];
  N73 -> Q1;
  Q2[label="Query"];
  N91 -> Q2;
  Q3[label="Query"];
  N98 -> Q3;
  Q4[label="Query"];
  N102 -> Q4;
  Q5[label="Query"];
  N107 -> Q5;
  Q6[label="Query"];
  N119 -> Q6;
  Q7[label="Query"];
  N120 -> Q7;
  Q8[label="Query"];
  N128 -> Q8;
  Q9[label="Query"];
  N134 -> Q9;
  Q10[label="Query"];
  N142 -> Q10;
  Q11[label="Query"];
  N149 -> Q11;
  Q12[label="Query"];
  N2 -> Q12;
  Q13[label="Query"];
  N156 -> Q13;
  Q14[label="Query"];
  N157 -> Q14;
  Q15[label="Query"];
  N164 -> Q15;
}
"""
        observed = g.to_dot()
        self.assertEqual(expected.strip(), observed.strip())
        self.assertEqual(len(queries), len(q))
        # Query node Q12 is the query on prevalence()
        self.assertEqual(12, q[prevalence()])
