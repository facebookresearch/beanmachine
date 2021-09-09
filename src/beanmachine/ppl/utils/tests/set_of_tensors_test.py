# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.utils.set_of_tensors import SetOfTensors
from torch import tensor


class SetOfTensorsTest(unittest.TestCase):
    def test_set_of_tensors_1(self) -> None:
        self.maxDiff = None

        # Show that we deduplicate these tensors.

        t = [
            tensor(1.0),
            tensor([]),
            tensor([1.0]),
            tensor([1.0, 2.0]),
            tensor([1.0, 2.0, 3.0, 4.0]),
            tensor([[1.0]]),
            tensor([[1.0], [2.0]]),
            tensor([[1.0, 2.0]]),
            tensor([[1.0, 2.0], [3.0, 4.0]]),
            tensor(1.0),
            tensor([]),
            tensor([1.0]),
            tensor([1.0, 2.0]),
            tensor([1.0, 2.0, 3.0, 4.0]),
            tensor([[1.0]]),
            tensor([[1.0], [2.0]]),
            tensor([[1.0, 2.0]]),
            tensor([[1.0, 2.0], [3.0, 4.0]]),
        ]

        s = SetOfTensors(t)

        self.assertEqual(9, len(s))

        observed = "\n".join(sorted(str(i) for i in s))
        expected = """
tensor(1.)
tensor([1., 2., 3., 4.])
tensor([1., 2.])
tensor([1.])
tensor([[1., 2.],
        [3., 4.]])
tensor([[1., 2.]])
tensor([[1.],
        [2.]])
tensor([[1.]])
tensor([])"""
        self.assertEqual(expected.strip(), observed.strip())
