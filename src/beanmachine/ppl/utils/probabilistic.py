# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
from functools import wraps

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import BMGNode


def probabilistic(bmg: BMGraphBuilder):
    """
    Decorator to be used to make sample function probabilistic
    """

    def inner(f):
        @wraps(f)
        def wrapper(*args):
            for i in range(len(args)):
                arg = args[i]
                if isinstance(arg, BMGNode):
                    key_value_pairs = []
                    for key in arg.support():
                        newargs = list(args)
                        newargs[i] = key
                        value = wrapper(*newargs)
                        key_value_pairs.append(bmg.add_constant(key))
                        key_value_pairs.append(value)
                    map_node = bmg.add_map(*key_value_pairs)
                    index_node = bmg.add_index(map_node, arg)
                    return index_node
            # If we got here, there were no distribution arguments.
            return f(*args)

        if inspect.ismethod(f):
            meth_name = f.__name__ + "_wrapper"
            setattr(f.__self__, meth_name, wrapper)
        else:
            f._wrapper = wrapper
        return wrapper

    return inner
