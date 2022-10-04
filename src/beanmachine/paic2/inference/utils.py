import ast
import sys
import typing


def to_name(func) -> str:
    # TODO: check globals
    if isinstance(func, ast.Name):
        return func.id
    path = []
    current = func
    while isinstance(current, ast.Attribute):
        path.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        path.append(current.id)
    full_name = ""
    i = 0
    for segment in path.__reversed__():
        if i == 0:
            full_name = segment
        else:
            full_name = full_name + "." + segment
        i = i + 1
    return full_name


def get_globals(f: typing.Callable) -> typing.Dict:
    if f.__module__ not in sys.modules:
        msg = (
            f"module {f.__module__} for function {f.__name__} not "
            + f"found in sys.modules.\n{str(sys.modules.keys())}"
        )
        raise Exception(msg)
    return sys.modules[f.__module__].__dict__
