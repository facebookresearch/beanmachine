# Copyright (c) Facebook, Inc. and its affiliates.
"""Error reporting for internal compiler errors"""

from ast import AST
from typing import Optional


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

        message = f"""Compilation of the lifted AST failed.
This typically indicates an internal error in the rewrite phase of the compiler.
### Exception thrown ###
{original_exception}
"""

        InternalError.__init__(self, message, original_exception)
