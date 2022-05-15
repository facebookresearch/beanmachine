# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Set

from beanmachine.ppl.compiler.bmg_nodes import BMGNode
from beanmachine.ppl.utils.multidictionary import MultiDictionary

# We wish to associate the program state at the time of node creation
# with that node, so that we can produce better diagnostics, error messages,
# and so on.


class FunctionCall:
    # A record of a particular function call.
    func: Callable
    args: Any
    kwargs: Dict[str, Any]

    def __init__(self, func: Callable, args: Any, kwargs: Dict[str, Any]) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        func = self.func.__name__
        args = ",".join(str(arg) for arg in self.args)
        kwargs = ",".join(
            sorted(
                (str(kwarg) + "=" + str(self.kwargs[kwarg])) for kwarg in self.kwargs
            )
        )
        comma = "," if len(args) > 0 and len(kwargs) > 0 else ""
        return f"{func}({args}{comma}{kwargs})"


class CallStack:
    _stack: List[FunctionCall]

    def __init__(self) -> None:
        self._stack = []

    def push(self, FunctionCall: FunctionCall) -> None:
        self._stack.append(FunctionCall)

    def pop(self) -> FunctionCall:
        return self._stack.pop()

    def peek(self) -> Optional[FunctionCall]:
        return self._stack[-1] if len(self._stack) > 0 else None


_empty_kwargs = {}


class ExecutionContext:
    # An execution context does two jobs right now:
    # * Tracks the current call stack
    # * Maintains a map from nodes to the function call that
    #   created them.
    #
    # NOTE: Because most nodes are deduplicated, it is possible that
    # one node is associated with multiple calls. We therefore have
    # a multidictionary that maps from nodes to a set of function calls.

    _stack: CallStack
    _node_locations: MultiDictionary  # BMGNode -> {FunctionCall}

    def __init__(self) -> None:
        self._stack = CallStack()
        self._node_locations = MultiDictionary()

    def current_site(self) -> Optional[FunctionCall]:
        return self._stack.peek()

    def record_node_call(
        self, node: BMGNode, site: Optional[FunctionCall] = None
    ) -> None:
        if site is None:
            site = self.current_site()
        if site is not None:
            self._node_locations.add(node, site)

    def node_locations(self, node: BMGNode) -> Set[FunctionCall]:
        return self._node_locations[node]

    def call(
        self, func: Callable, args: Any, kwargs: Dict[str, Any] = _empty_kwargs
    ) -> Any:
        self._stack.push(FunctionCall(func, args, kwargs))
        try:
            return func(*args, **kwargs)
        finally:
            self._stack.pop()
