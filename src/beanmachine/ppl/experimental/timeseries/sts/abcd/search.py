# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import dill
import gpytorch as gp
import torch
import torch.multiprocessing as mp
from gpytorch.kernels import Kernel
from sts.abcd.expansion import expand_kernel, initialize_kernel, simplify
from sts.abcd.expression import KernelExpression
from sts.abcd.model import ABCDExactGPModel
from sts.abcd.utils import GRAMMAR_RULES, is_kernel_type_eq, remove_redundancy
from sts.data import DataTensor
from sts.gp.kernels import WhiteNoiseKernel


"""
This file is runnable only locally if run_in_parallel is True, not using buck.
"""


class Search(object):
    """
    :param x_train: input training data.
    :param y_train: output training data.
    """

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        self.x_train = x_train
        self.y_train = y_train

    def train_model(
        self,
        candidate_kernel: Kernel,
        learning_rate: float,
        num_epochs: int,
        method: str,
    ) -> float:
        """
        Train a model given the kernel to to optimize params and evaluate the score.
        :param candidate_kernel: the candidate kernel.
        :param learning_rate: learning rate for training models.
        :param num_epochs: number of epochs for training models.
        :param method: method for scoring, one of 'BIC', 'AIC', 'NLL'.
        :return: the score of the trained model.
        """
        likelihood = gp.likelihoods.GaussianLikelihood()
        x_train = DataTensor(self.x_train, header=["x"])
        y_train = DataTensor(self.y_train, header=["y"])
        model = ABCDExactGPModel(x_train, y_train, likelihood, candidate_kernel)
        model.optimize_params(learning_rate=learning_rate, num_epochs=num_epochs)
        score = model.score(method=method).item()
        return score

    def _get_all_candidate_kernels(
        self,
        best_kernels: List[Tuple[Kernel, float]],
        grammar: Dict,
        period_value_list: Tuple[float],
        depth: int,
    ) -> Tuple[List[Kernel], str]:
        """
        Get all candidate kernels by operating all possible operators.
        :param best_kernels: the list of kernels to expand from.
        :param grammar: the grammar operator rules for expanding current kernel to neighbor kernels.
        :param peiord_value_list: the tuple of possible values of period lengths.
        :param depth: the depth of search.
        :return: a list of kernels and output information.
        """
        nb_kernels = []
        for best_kernel, _ in best_kernels:
            nb_kernels.extend(expand_kernel(best_kernel, grammar))
        nb_kernels = remove_redundancy(nb_kernels)
        lines = f"before simplify={len(nb_kernels)}\n"

        # simplify kernel structures and remove redundant kernels
        sim_kernels = []
        for kern in nb_kernels:
            sim_kernels.append(simplify(kern))
        nb_kernels = remove_redundancy(sim_kernels)
        lines += f"after simplify={len(nb_kernels)}\n"

        # add heuristic restart of period length for periodic kenerls
        restart_period_kernels = []
        for kern in nb_kernels:
            restart_period_kernels.extend(initialize_kernel(kern, period_value_list))
        nb_kernels.extend(restart_period_kernels)

        lines += f"after period restart={len(nb_kernels)}\n"

        lines += f"number of candidate kernels at depth {depth} is {len(nb_kernels)}\n"
        return nb_kernels, lines

    def search(
        self,
        max_depth: int,
        top_k: int,
        period_value_list: Tuple[float],
        output_file: str = "./search.txt",
        pickle_file: Optional[str] = "./result.pkl",
        grammar: List[str] = GRAMMAR_RULES["all"],
        learning_rate: float = 0.01,
        num_epochs: int = 1000,
        method: str = "BIC",
        run_in_parallel: bool = False,
        pool_num: int = 4,
    ) -> List[Tuple[Kernel, float]]:
        """
        Search algorithm to find the best kernel given the training dataset
        with regard to BIC score.
        We keep k best models in each depth and end the search when best models
        are not updated or reach the preset maximum depth.
        At each depth n, we do the following:
            1. expand the kernel of best models by our operators;
            2. simplify kernel structures and remove redundant kernels;
            3. add heuristic restart of period length for periodic kenerls;
            4. train each model with a kernel and evaluate with BIC score.
            5. update k best kernels.
        If any of best kernels are updated or depth hasn't reach the maximum, we go back to step 1 and continue the loop for depth n+1.
        Search method to search the kernel. This is the main function for ABCD method.

        :param max_depth: maximum number of depth to search.
        :param top_k: keep top k models in each depth.
        :param period_value_list: the tuple of period length values.
        :param output_file: output file path.
        :param pickle_file: result model file path.
        :param grammar: list of representations for operators to use.
        :param learning_rate: learning rate for training models.
        :param num_epochs: number of epochs for training models.
        :param method: method for scoring, one of 'BIC', 'AIC', 'NLL'.
        :param run_in_parallel: if true, run the training of model in parallel.
        :param pool_num: the number of pools for multiprocessing.
        :return: a list of kernels and their scores.
        """

        with open(output_file, "w") as file:
            start_time = time.time()
            # initialize
            current_kernel = WhiteNoiseKernel(noise=1e-4)
            model_score = self.train_model(
                current_kernel,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                method=method,
            )
            lines = f"score={model_score}\n"
            best_kernels = [(current_kernel, model_score)]

            lines += "--------search begin--------\n"

            for depth in range(max_depth):
                # expand the kernel of best models
                nb_kernels, lines = self._get_all_candidate_kernels(
                    best_kernels, grammar, period_value_list, depth
                )

                # train each model with a kernel and evaluate with score
                score_list = []
                train_partial = partial(
                    self.train_model,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    method=method,
                )

                if run_in_parallel:
                    with mp.Pool(pool_num) as pool:
                        score_list = pool.map(train_partial, nb_kernels)
                else:
                    score_list = map(train_partial, nb_kernels)

                kernel_score_list = list(zip(nb_kernels, score_list))

                kernel_score_list.extend(best_kernels)
                kernel_score_list.sort(key=lambda x: x[1])
                best_kernels_new = kernel_score_list[:top_k]

                best_fixed = True
                for i in range(top_k):
                    if not is_kernel_type_eq(
                        best_kernels_new[i][0], best_kernels[i][0]
                    ):
                        best_fixed = False
                        break
                if best_fixed:
                    break
                best_kernels = best_kernels_new

                lines += "--------depth=" + str(depth) + "--------\n"
                for k, score in kernel_score_list:
                    lines += repr(KernelExpression(k)) + ";" + str(score) + "\n"
                lines += "--------best kernel--------\n"
                for best_kernel, score in best_kernels:
                    lines += (
                        repr(KernelExpression(best_kernel)) + ";" + str(score) + "\n"
                    )
                file.write(lines)
                file.flush()
                lines = ""

            lines += "--------overall depth=" + str(depth) + "--------\n"
            for best_kernel, score in best_kernels:
                lines += repr(KernelExpression(best_kernel)) + "\n"
                lines += str(score) + "\n"
            lines += "--- %s seconds ---" % (time.time() - start_time)
            file.write(lines)
            file.close()
        if pickle_file is not None:
            with open(pickle_file, "wb") as handle:
                dill.dump(best_kernels, handle)
        return best_kernels
