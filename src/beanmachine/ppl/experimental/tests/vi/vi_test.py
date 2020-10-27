# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.experimental.vi.VariationalInfer import VariationalInfer
from beanmachine.ppl.model.statistical_model import StatisticalModel

from .neals_funnel import NealsFunnel


class VariationalInferTest(unittest.TestCase):
    def tearDown(self) -> None:
        StatisticalModel.reset()

    def test_neals_funnel(self):
        nf = NealsFunnel()

        vi = VariationalInfer(target=nf)
        vi.train(epochs=300)
