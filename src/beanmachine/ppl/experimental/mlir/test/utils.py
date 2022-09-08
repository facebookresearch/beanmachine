import ast
import typing
import sys

def _unindent(lines):
    # TODO: Handle the situation if there are tabs
    if len(lines) == 0:
        return lines
    num_spaces = len(lines[0]) - len(lines[0].lstrip(" "))
    if num_spaces == 0:
        return lines
    spaces = lines[0][0:num_spaces]
    return [(line[num_spaces:] if line.startswith(spaces) else line) for line in lines]

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

def _get_globals(f: typing.Callable) -> typing.Dict:
    """Given a function, returns the global variable dictionary of the
    module containing the function."""

    if f.__module__ not in sys.modules:
        msg = (
            f"module {f.__module__} for function {f.__name__} not "
            + f"found in sys.modules.\n{str(sys.modules.keys())}"
        )
        raise Exception(msg)
    return sys.modules[f.__module__].__dict__