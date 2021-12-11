# @lint-ignore-every LICENSELINT
# Sourced/adapted from: https://github.com/pyro-ppl/pyro
#
# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Type,
    Union,
)
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Optimizer

from ...model.rv_identifier import RVIdentifier


_MODULE_NAMESPACE_DIVIDER = "$$$"


def normalize_param_name(name):
    return name.replace(_MODULE_NAMESPACE_DIVIDER, ".")


def module_from_param_with_module_name(param_name):
    return param_name.split(_MODULE_NAMESPACE_DIVIDER)[0]


def user_param_name(param_name):
    if _MODULE_NAMESPACE_DIVIDER in param_name:
        return param_name.split(_MODULE_NAMESPACE_DIVIDER)[1]
    return param_name


class MultiOptimizer:
    """
    Base class of optimizers that make use of higher-order derivatives.
    Higher-order optimizers generally use :func:`torch.autograd.grad` rather
    than :meth:`torch.Tensor.backward`, and therefore require a different
    interface from usual optimizers. In this interface, the :meth:`step`
    method inputs a ``loss`` tensor to be differentiated, and backpropagation
    is triggered one or more times inside the optimizer. Derived classes must
    implement :meth:`step` to compute derivatives and update parameters
    in-place.
    """

    def step(self, loss: torch.Tensor, params: Dict) -> None:
        """
        Performs an in-place optimization step on parameters given a
        differentiable ``loss`` tensor.
        Note that this detaches the updated tensors.
        :param torch.Tensor loss: A differentiable tensor to be minimized.
            Some optimizers require this to be differentiable multiple times.
        :param dict params: A dictionary mapping param name to unconstrained
            value as stored in the param store.
        """
        updated_values = self.get_step(loss, params)
        for name, value in params.items():
            with torch.no_grad():
                # we need to detach because updated_value may depend on value
                value.copy_(updated_values[name].detach())

    def get_step(self, loss: torch.Tensor, params: Dict) -> Dict:
        """
        Computes an optimization step of parameters given a differentiable
        ``loss`` tensor, returning the updated values.
        Note that this preserves derivatives on the updated tensors.
        :param torch.Tensor loss: A differentiable tensor to be minimized.
            Some optimizers require this to be differentiable multiple times.
        :param dict params: A dictionary mapping param name to unconstrained
            value as stored in the param store.
        :return: A dictionary mapping param name to updated unconstrained
            value.
        :rtype: dict
        """
        raise NotImplementedError


class BMMultiOptimizer(MultiOptimizer):
    """
    Facade to wrap :class:`BMOptim` objects in a :class:`MultiOptimizer`
    interface.
    """

    def __init__(self, optim: "BMOptim") -> None:
        if not isinstance(optim, BMOptim):
            raise TypeError(
                "Expected a BMOptim object but got a {}".format(type(optim))
            )
        self.optim = optim

    def step(
        self, loss: torch.Tensor, params: Dict[RVIdentifier, nn.Parameter]
    ) -> None:
        values = params.values()
        grads = torch.autograd.grad(loss, values, create_graph=True)
        for x, g in zip(values, grads):
            x.grad = g
        self.optim(params)


class BMOptim:
    """
    A wrapper for torch.optim.Optimizer objects that helps with managing dynamically generated parameters.

    :param optim_constructor: a torch.optim.Optimizer
    :param optim_args: a dictionary of learning arguments for the optimizer or a callable that returns
        such dictionaries
    :param clip_args: a dictionary of clip_norm and/or clip_value args or a callable that returns
        such dictionaries
    """

    def __init__(
        self,
        optim_constructor: Union[Callable, Optimizer, Type[Optimizer]],
        optim_args: Union[Dict, Callable[..., Dict]],
        clip_args: Optional[Union[Dict, Callable[..., Dict]]] = None,
    ):
        self.pt_optim_constructor = optim_constructor

        # must be callable or dict
        assert callable(optim_args) or isinstance(
            optim_args, dict
        ), "optim_args must be function that returns defaults or a defaults dictionary"

        if clip_args is None:
            clip_args = {}

        # must be callable or dict
        assert callable(clip_args) or isinstance(
            clip_args, dict
        ), "clip_args must be function that returns defaults or a defaults dictionary"

        # hold our args to be called/used
        self.pt_optim_args = optim_args
        if callable(optim_args):
            self.pt_optim_args_argc = len(inspect.signature(optim_args).parameters)
        self.pt_clip_args = clip_args

        # holds the torch optimizer objects
        self.optim_objs: Dict = {}
        self.grad_clip: Dict = {}

        # any optimizer state that's waiting to be consumed (because that parameter hasn't been seen before)
        self._state_waiting_to_be_consumed: Dict = {}

    def __call__(
        self, params: Dict[RVIdentifier, nn.Parameter], *args, **kwargs
    ) -> None:
        """
        :param params: a list of parameters
        :type params: an iterable of strings

        Do an optimization step for each param in params. If a given param has never been seen before,
        initialize an optimizer for it.
        """
        for p, param_tensor in params.items():
            # if we have not seen this param before, we instantiate an optim object to deal with it
            if p not in self.optim_objs:
                # create a single optim object for that param
                self.optim_objs[p] = self._get_optim(param_tensor)
                # create a gradient clipping function if specified
                self.grad_clip[p] = self._get_grad_clip(param_tensor)
                # set state from _state_waiting_to_be_consumed if present
                param_name = str(p)
                if param_name in self._state_waiting_to_be_consumed:
                    state = self._state_waiting_to_be_consumed.pop(param_name)
                    self.optim_objs[p].load_state_dict(state)

            if self.grad_clip[p] is not None:
                self.grad_clip[p](p)

            if isinstance(
                self.optim_objs[p], torch.optim.lr_scheduler._LRScheduler
            ) or isinstance(
                self.optim_objs[p], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                # if optim object was a scheduler, perform an optimizer step
                self.optim_objs[p].optimizer.step(*args, **kwargs)
            else:
                self.optim_objs[p].step(*args, **kwargs)

    def get_state(self) -> Dict:
        """
        Get state associated with all the optimizers in the form of a dictionary with
        key-value pairs (parameter name, optim state dicts)
        """
        state_dict = {}
        for param in self.optim_objs:
            param_name = str(param)
            state_dict[param_name] = self.optim_objs[param].state_dict()
        return state_dict

    def set_state(self, state_dict: Dict) -> None:
        """
        Set the state associated with all the optimizers using the state obtained
        from a previous call to get_state()
        """
        self._state_waiting_to_be_consumed = state_dict

    def save(self, filename: str) -> None:
        """
        :param filename: file name to save to
        :type filename: str

        Save optimizer state to disk
        """
        with open(filename, "wb") as output_file:
            torch.save(self.get_state(), output_file)

    def load(self, filename: str) -> None:
        """
        :param filename: file name to load from
        :type filename: str

        Load optimizer state from disk
        """
        with open(filename, "rb") as input_file:
            state = torch.load(input_file)
        self.set_state(state)

    def _get_optim(self, param: Union[Iterable[Tensor], Iterable[Dict[Any, Any]]]):
        return self.pt_optim_constructor([param], **self._get_optim_args(param))  # type: ignore

    # helper to fetch the optim args if callable (only used internally)
    def _get_optim_args(self, param: Union[Iterable[Tensor], Iterable[Dict]]):
        # If we were passed a function, we call function with a
        # fully qualified name e.g. 'mymodule.mysubmodule.bias'.
        if callable(self.pt_optim_args):
            pt_optim_args = cast(Callable, self.pt_optim_args)
            param_name = str(param)
            if self.pt_optim_args_argc == 1:
                # Normalize to the format of nn.Module.named_parameters().
                normal_name = normalize_param_name(param_name)
                opt_dict = pt_optim_args(normal_name)
            else:
                # DEPRECATED Split param name in to pieces.
                module_name = module_from_param_with_module_name(param_name)
                stripped_param_name = user_param_name(param_name)
                opt_dict = pt_optim_args(module_name, stripped_param_name)

            # must be dictionary
            assert isinstance(
                opt_dict, dict
            ), "per-param optim arg must return defaults dictionary"
            return opt_dict
        else:
            return self.pt_optim_args

    def _get_grad_clip(self, param: Union[Iterable[Tensor], Iterable[Dict]]):
        grad_clip_args = self._get_grad_clip_args(param)

        if not grad_clip_args:
            return None

        def _clip_grad(params: Union[Tensor, Iterable[Tensor]]):
            self._clip_grad(params, **grad_clip_args)

        return _clip_grad

    def _get_grad_clip_args(
        self, param: Union[Iterable[Tensor], Iterable[Dict]]
    ) -> Dict:
        # if we were passed a fct, we call fct with param info
        # arguments are (module name, param name) e.g. ('mymodule', 'bias')
        if callable(self.pt_clip_args):
            pt_clip_args = cast(Callable, self.pt_clip_args)

            # get param name
            param_name = str(param)
            module_name = module_from_param_with_module_name(param_name)
            stripped_param_name = user_param_name(param_name)

            # invoke the user-provided callable
            clip_dict = pt_clip_args(module_name, stripped_param_name)

            # must be dictionary
            assert isinstance(
                clip_dict, dict
            ), "per-param clip arg must return defaults dictionary"
            return clip_dict
        else:
            pt_clip_args = cast(Dict, self.pt_clip_args)
            return pt_clip_args

    @staticmethod
    def _clip_grad(
        params: Union[Tensor, Iterable[Tensor]],
        clip_norm: Optional[Union[int, float]] = None,
        clip_value: Optional[Union[int, float]] = None,
    ) -> None:
        if clip_norm is not None:
            clip_grad_norm_(params, clip_norm)
        if clip_value is not None:
            clip_grad_value_(params, clip_value)
