# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import inspect
import pkgutil
from typing import Dict, Mapping, Sequence, Tuple, Callable, Optional, Any
from types import ModuleType
from collections import OrderedDict
from inspect import isclass, isfunction, ismodule

def documentable_symbols(module: ModuleType) -> Sequence[Tuple[str, Any]]:
    """
    Given a module object, returns a list of object name and values for documentable
    symbols (functions and classes defined in this module or a subclass)
    """
    return [
        (n, m) for n, m in inspect.getmembers(module, None)
        if isfunction(m) or (isclass(m) and m.__module__.startswith(module.__name__))
        ]

def walk_packages(
    modname: str, filter: Optional[Callable[[Any], bool]] = None
) -> Mapping[str, Tuple[ModuleType, Sequence[Tuple[str, Any]]]]:
    """
    Given a base module name, return a mapping from the name of all modules
    accessible under the base to a tuple of module and symbol objects.

    A symbol is represented by a tuple of the object name and value, and is
    either a function or a class accessible when the module is imported.

    """
    module = importlib.import_module(modname)
    modules = {modname: (module, documentable_symbols(module))}

    # NOTE: I use path of flowtorch rather than e.g. flowtorch.bijectors
    # to avoid circular imports
    path = module.__path__  # type: ignore

    # The followings line uncovered a bug that hasn't been fixed in mypy:
    # https://github.com/python/mypy/issues/1422
    for importer, this_modname, _ in pkgutil.walk_packages(
        path=path,  # type: ignore  # mypy issue #1422
        prefix=f"{module.__name__}.",
        onerror=lambda x: None,
    ):
        # Conditions required for mypy
        if importer is not None:
            if isinstance(importer, importlib.abc.MetaPathFinder):
                finder = importer.find_module(this_modname, None)
            elif isinstance(importer, importlib.abc.PathEntryFinder):
                finder = importer.find_module(this_modname)
        else:
            finder = None

        if finder is not None:
            module = finder.load_module(this_modname)

        else:
            raise Exception("Finder is none")

        if module is not None:
            # Get all classes and functions imported/defined in module
            """symbols = [
                (n, m) for n, m in inspect.getmembers(module, None)
                        if isfunction(m) or (isclass(m) and m.__module__.startswith(module.__name__))
            ]"""

            modules[this_modname] = (module, documentable_symbols(module))

            del module
            del finder

        else:
            raise Exception("Module is none")

    return modules

def sparse_module_hierarchy(mod_names: Sequence[str]) -> Mapping[str, Any]:
    # Make list of modules to search and their hierarchy, pruning entries that
    # aren't in mod_names
    results: Dict[str, Any] = OrderedDict()
    this_dict = results

    for module in sorted(mod_names):
        submodules = module.split(".")

        # Navigate to the correct insertion place for this module
        for idx in range(0, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            if submodule in this_dict:
                this_dict = this_dict[submodule]

        # Insert module if it doesn't exist already
        this_dict.setdefault(module, {})

    return results