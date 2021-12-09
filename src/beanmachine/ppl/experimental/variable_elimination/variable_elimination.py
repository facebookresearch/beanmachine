# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from beanmachine.ppl.experimental.variable_elimination.util import (
    dict_from_first_to_second_component,
)

try:
    from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
        PyTorchTableFactor,
    )
    from neuralpp.inference.graphical_model.variable.integer_variable import (
        IntegerVariable,
    )

    def make_neuralpp_model(bm_model_info):
        """
        Makes a neuralpp model with information from
        a BM model in the following format:
        bm_model_info ::= discrete_variable_definition+
        discrete_variable_definition ::= ( var_id, support_size, all_var_ids_in_function, log_probs )
        where var_id is the variable identifier of the random variable being defined,
        support_size is the number of values of this variable,
        all_var_ids_in_function: an iterable of variable identifiers of all random variables in
            the defined random variable's function.
        log_probs: an iterable over the log prob values of the distribution returned by the function
            for each assignment to the random variables,
            where assignments are ordered by iterating variables in all_var_ids_in_function,
            with the last variable being iterated most frequently.
        """

        support_sizes_dict = dict_from_first_to_second_component(bm_model_info)

        model = [
            make_neuralpp_factor(all_var_ids_in_function, support_sizes_dict, log_probs)
            for _, _, all_var_ids_in_function, log_probs in bm_model_info
        ]

        return model

    def make_neuralpp_factor(all_var_ids_in_function, support_sizes_dict, log_probs):
        support_sizes_in_function = [
            support_sizes_dict[var_id] for var_id in all_var_ids_in_function
        ]
        variables = [
            IntegerVariable(str(var_id), support_size)
            for (var_id, support_size) in zip(
                all_var_ids_in_function, support_sizes_in_function
            )
        ]
        tensor_of_log_probs = torch.tensor(log_probs).reshape(
            *support_sizes_in_function
        )
        return PyTorchTableFactor(variables, tensor_of_log_probs, log_space=True)

    class BMBasedNeuralPPModel:
        def __init__(self, bm_model_info):
            """
            Initializes BMBasedNeuralPPModel object with information from
            a BM model as defined in make_neuralpp_model.
            """
            self.model = self.make_neuralpp_model(bm_model_info)


except ImportError:
    pass
