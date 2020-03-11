# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple


class PPLBenchPPL(object, metaclass=ABCMeta):
    @abstractmethod
    def obtain_posterior(self, data_train, args_dict, model=None) -> Tuple[List, Dict]:
        """
        Returns posterior samples and timing info
        :param data_train: data to train the model on
        :param args_dict: model specific parameters like number of observations etc

        :returns: Tuple of
        1) posterior_samples: List where each element is a dictionary with key as
        name of random variable and value as sample.
        2) Timing Info: {"compile_time": ..., "inference_time": ...}
        """
        raise NotImplementedError("PPLBenchPPL must implement obtain_posterior.")
