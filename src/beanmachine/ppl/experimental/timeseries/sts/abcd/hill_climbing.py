# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random
import time
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Tuple

import dill
import torch
import torch.multiprocessing as mp
from gpytorch.kernels import Kernel
from sts.abcd.expansion import expand_kernel
from sts.abcd.expression import KernelExpression
from sts.abcd.search import Search
from sts.abcd.utils import BASE_KERNELS, GRAMMAR_RULES


"""
This file is runnable only locally if run_in_parallel is True, not using buck.
The result is showing in notebook N1007190.
"""


class HillClimbingSearch(Search):
    """
    :param x_train: input training data.
    :param y_train: output training data.
    """

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        super().__init__(x_train, y_train)

    def _get_random_neighbor(self, kernel: Kernel, rules: Dict) -> Kernel:
        """
        Get a random neighbor kernel of the current kernel
        :param kernel: current kernel to get neighbor.
        :param rules: the grammar rules that can be used to expand the current kernel.
        :return: a neighbor kernel.
        """
        nb_kernels = []
        while len(nb_kernels) == 0:
            rule = random.choices(rules, k=1)
            nb_kernels = expand_kernel(kernel, rule)
        return random.choices(nb_kernels, k=1)[0]

    def _search_one_round(
        self,
        kernel: Kernel,
        score: float,
        rules: Dict,
        learning_rate: float,
        num_epochs: int,
        method: str,
        reject: int,
    ) -> Tuple[Kernel, int, int]:
        """
        One iteration of the search to get one neighbor kernel and train, decide whether or not to accept the new kernel by the scores.
        :param kernel: current kernel to get neighbor.
        :param score: the score of current kernel.
        :param rules: the grammar rules that can be used to expand the current kernel.
        :param learning_rate: learning rate of training GP model.
        :param num_epochs: the number of epochs in traning GP model.
        :param method: scoring method of kernels, one of 'AIC', 'BIC' or 'NLL'.
        :param reject: the count of rejections.
        :return: a better kernel, which is the neighbor kernel if its score is better or the current kernel otherwise,
                 the score of the returned kernel,
                 and the count of rejections.
        """
        nb_kernel = self._get_random_neighbor(kernel, rules)
        score_new = self.train_model(nb_kernel, learning_rate, num_epochs, method)
        if score_new < score:
            return nb_kernel, score_new, reject
        else:
            return kernel, score, reject + 1

    def _search_one_start(
        self,
        restart_id: int,
        num_iters: int,
        grammar: Dict,
        top_k: int,
        learning_rate: float,
        num_epochs: int,
        method: str,
    ) -> Tuple[List[Tuple[Kernel, float]], str]:
        """
        One hill climbing search process.
        :param restart_id: the id of this search, search with different id has different start.
        :param num_iters: the number of iterations in this search.
        :param grammar: the grammar rules that can be used to expand kernels.
        :param top_k: return top k kernels.
        :param learning_rate: learning rate of training.
        :param num_epochs: the number of epochs in traning.
        :param method: scoring method of kernels, one of 'AIC', 'BIC' or 'NLL'.
        :return: the kernel, score pair of the top k kernels, and the string info to write to file.
        """
        kernel_list = random.choices(BASE_KERNELS, k=2)
        best_kernel = deepcopy(kernel_list[0]) + deepcopy(kernel_list[1])
        score = self.train_model(
            best_kernel,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            method=method,
        )
        best_kernels = []
        best_kernels.append((best_kernel, score))
        reject_cnt = 0

        for _ in range(num_iters):
            best_kernel, score, reject_cnt_new = self._search_one_round(
                best_kernel,
                score,
                grammar,
                learning_rate,
                num_epochs,
                method,
                reject_cnt,
            )
            if reject_cnt == reject_cnt_new:
                best_kernels.append((best_kernel, score))
                reject_cnt = reject_cnt_new
            if reject_cnt >= 0.6 * num_iters:
                break

        best_kernels.sort(key=lambda x: x[1])
        lines = f"--------Round {restart_id}--------\n"
        for best_kernel, score in best_kernels[:top_k]:
            lines += repr(KernelExpression(best_kernel)) + ";" + str(score) + "\n"

        return best_kernels[:top_k], lines

    def search(
        self,
        num_iters: int,
        num_restarts: int,
        top_k: int,
        output_file: str = "./search.txt",
        pickle_file: Optional[str] = "./result.pkl",
        grammar: List[str] = GRAMMAR_RULES["all"],
        learning_rate: float = 0.01,
        num_epochs: int = 1000,
        method: str = "BIC",
        run_in_parallel: bool = False,
    ) -> List[Tuple[Kernel, float]]:
        """
        The search with several restarts.
        :param num_iters: the number of iterations in this search.
        :param num_restarts: the number of restarts, each restart is an independent search chain.
        :param top_k: return top k kernels.
        :param output_file: the path of the output file.
        :param pickle_file: the path of the file to pickle the best kernels.
        :param grammar: the grammar rules that can be used to expand kernels.
        :param learning_rate: learning rate of training.
        :param num_epochs: the number of epochs in traning.
        :param method: scoring method of kernels, one of 'AIC', 'BIC' or 'NLL'.
        :return: the kernel, score pair of the top k kernels.
        """
        with open(output_file, "w") as file:
            start_time = time.time()

            lines = "--------search begin---------\n"
            all_good_kernels = []

            search_partial = partial(
                self._search_one_start,
                num_iters=num_iters,
                grammar=grammar,
                top_k=top_k,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                method=method,
            )
            if run_in_parallel:
                with mp.Pool(num_restarts) as pool:
                    result_list = pool.map(search_partial, range(num_restarts))
            else:
                result_list = map(search_partial, range(num_restarts))
            for kernels, ss in result_list:
                all_good_kernels.extend(kernels)
                lines += ss
            all_good_kernels.sort(key=lambda x: x[1])
            lines += "--------overall Result--------\n"
            for best_kernel, score in all_good_kernels[:top_k]:
                lines += repr(KernelExpression(best_kernel)) + ";" + str(score) + "\n"
            lines += "--- %s seconds ---" % (time.time() - start_time)
            file.write(lines)
            file.close()
        if pickle_file is not None:
            with open(pickle_file, "wb") as handle:
                dill.dump(all_good_kernels[:top_k], handle)
        return all_good_kernels[:top_k]
