# Copyright (c) Facebook, Inc. and its affiliates.
"""Error reporting for internal compiler errors"""

import os
from ast import AST
from typing import Optional

import beanmachine.ppl.compiler.ast_tools as ast_tools


_BEANSTALK_VERBOSE_EXCEPTIONS = "BEANSTALK_VERBOSE_EXCEPTIONS"

_help_verbose = f"""Set environment variable {_BEANSTALK_VERBOSE_EXCEPTIONS}
to 1 for extended error information.
"""


def _check_environment(variable: str) -> bool:
    return os.environ.get(variable) == "1"


# You can change this to True for debugging purposes.
_always_verbose_exceptions = False


def _verbose_exceptions() -> bool:
    return _always_verbose_exceptions or _check_environment(
        _BEANSTALK_VERBOSE_EXCEPTIONS
    )


class InternalError(Exception):
    """An exception class for internal compiler errors"""

    original_exception: Optional[Exception]

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        self.original_exception = original_exception
        Exception.__init__(self, message)


class LiftedCompilationError(InternalError):
    """An exception class for internal compiler errors when
compiling the lifted code."""

    source: str
    ast: AST

    def __init__(self, source: str, ast: AST, original_exception: Exception):
        self.source = source
        self.ast = ast

        # TODO: Consider adding a compiler version number, hash
        # or other identifier to help debug.

        brief = f"""Compilation of the lifted AST failed.
This typically indicates an internal error in the rewrite phase of the compiler.
### Exception thrown ###
{original_exception}
"""

        verbose = f"""### Internal compiler error ###
{brief}
### Model source ###
{source}
### Abstract syntax tree ###
from ast import *
failed = {ast_tools.print_python(ast)}
### End internal compiler error ###
"""

        use_verbose = _verbose_exceptions()

        help_text = "" if use_verbose else _help_verbose

        message = verbose if use_verbose else brief
        message += help_text

        InternalError.__init__(self, message, original_exception)
