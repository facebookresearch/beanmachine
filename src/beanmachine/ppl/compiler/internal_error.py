# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Error reporting for internal compiler errors"""

import os
from ast import AST
from tempfile import NamedTemporaryFile
from typing import Optional

import beanmachine.ppl.compiler.ast_tools as ast_tools


_BEANSTALK_LOG_ERRORS_TO_DISK = "BEANSTALK_LOG_ERRORS_TO_DISK"

_BEANSTALK_VERBOSE_EXCEPTIONS = "BEANSTALK_VERBOSE_EXCEPTIONS"


_help_log = f"""Set environment variable {_BEANSTALK_LOG_ERRORS_TO_DISK}
to 1 to dump extended error information to a temporary file.
"""

_help_verbose = f"""Set environment variable {_BEANSTALK_VERBOSE_EXCEPTIONS}
to 1 for extended error information.
"""


def _log_to_disk(message: str) -> str:
    temp = NamedTemporaryFile(prefix="beanstalk_ice_", delete=False, mode="wt")
    try:
        temp.write(message)
    finally:
        temp.close()
    return temp.name


def _check_environment(variable: str) -> bool:
    return os.environ.get(variable) == "1"


# You can change this to True for debugging purposes.
_always_log_errors_to_disk = False


def _log_errors_to_disk() -> bool:
    return _always_log_errors_to_disk or _check_environment(
        _BEANSTALK_LOG_ERRORS_TO_DISK
    )


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

        log = _log_errors_to_disk()
        use_verbose = _verbose_exceptions()

        help_text = "" if log else _help_log
        help_text += "" if use_verbose else _help_verbose

        message = verbose if use_verbose else brief
        message += help_text

        if log:
            logname = _log_to_disk(verbose)
            message += f"\nExtended error information logged to {logname}\n"

        InternalError.__init__(self, message, original_exception)
