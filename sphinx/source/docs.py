# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import inspect
import os
import pkgutil
import re
from collections import OrderedDict
from inspect import isclass, isfunction
from types import ModuleType
from typing import Dict, Mapping, Sequence, Tuple, Callable, Optional, Any

import toml


def documentable_symbols(module: ModuleType) -> Sequence[Tuple[str, Any]]:
    """
    Given a module object, returns a list of object name and values for documentable
    symbols (functions and classes defined in this module or a subclass)
    """
    return [
        (n, m)
        for n, m in inspect.getmembers(module, None)
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

    for module in sorted(mod_names):
        this_dict = results
        submodules = module.split(".")

        # Navigate to the correct insertion place for this module
        for idx in range(0, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            if submodule in this_dict:
                this_dict = this_dict[submodule]

        # Insert module if it doesn't exist already
        this_dict.setdefault(module, {})

    return results


# Generate 1 rst for each module
def save_rst(mod_name, sub_mod_names):
    rst = f"""{mod_name}
{"="*len(mod_name)}
"""

    if len(sub_mod_names):
        sub_mods = "\n   ".join(sub_mod_names)
        rst = (
            rst
            + f"""
Subpackages
-----------

.. toctree::
   :maxdepth: 4

   {sub_mods}
"""
        )

    rst = (
        rst
        + f"""
Module contents
---------------

.. automodule:: {mod_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    )
    return rst


def print_include_modules(
    config_path: str = "../../website/documentation.toml",
) -> None:
    # Load and validate configuration file
    import beanmachine

    config_path = os.path.join(beanmachine.__path__[0], config_path)
    config = toml.load(config_path)

    # Enumerate documentable symbols keyed by module
    modules_and_symbols = search_package(config)

    for k in sorted(modules_and_symbols.keys()):
        print(k)


def search_package(config):
    # Validate module name to document
    assert (
        "settings" in config
        and "search" in config["settings"]
        and (
            type(config["settings"]["search"]) is str
            or type(config["settings"]["search"]) is list
        )
    )

    # Construct regular expressions for includes and excludes
    # Default include/exclude rules
    patterns = {
        "include": {"modules": re.compile(r".+"), "symbols": re.compile(r".+")},
        "exclude": {"modules": re.compile(r""), "symbols": re.compile(r"")},
    }

    # Override rules based on configuration file
    if "filters" in config:
        filters = config["filters"]
        for clude, rules in filters.items():
            for rule, pattern in rules.items():
                if type(pattern) is list:
                    pattern = "|".join(pattern)
                patterns[clude][rule] = re.compile(pattern)

    # Read in all modules and symbols
    search = config["settings"]["search"]
    search = [search] if type(search) is str else search
    modules_and_symbols = {}
    for modname in set(search):
        modules_and_symbols = {**modules_and_symbols, **walk_packages(modname)}

    # Apply filtering
    # TODO: Would be slightly faster if we applied module filtering inside walk_packages
    tmp = {}
    for x, y in modules_and_symbols.items():
        if (
            patterns["include"]["modules"].fullmatch(x) is not None
            and patterns["exclude"]["modules"].fullmatch(x) is None
        ):

            new_y1 = [
                (a, b)
                for a, b in y[1]
                if patterns["include"]["symbols"].fullmatch(x + "." + a) is not None
                and patterns["exclude"]["symbols"].fullmatch(x + "." + a) is None
            ]

            tmp[x] = (y[0], new_y1)

    return tmp


def make_rst(config_path: str = "../../website/documentation.toml") -> None:
    # Load and validate configuration file
    import beanmachine

    config_path = os.path.join(beanmachine.__path__[0], config_path)
    config = toml.load(config_path)

    # Enumerate documentable symbols keyed by module
    modules_and_symbols = search_package(config)

    # Build hierarchy of modules and flatten
    hierarchy = sparse_module_hierarchy(modules_and_symbols.keys())
    modules = []

    def dfs(node):
        for h, v in node.items():
            modules.append(h)
            dfs(v)

    dfs(hierarchy)

    # Generate index.rst
    modules = [m for m in modules if m.count(".") < 3]
    modules = "\n   ".join(modules)
    index_rst = f""".. BeanMachine documentation master file, created by a \
tool by stefanwebb.

Welcome to BeanMachine's documentation!
=======================================

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   {modules}

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

    with open("index.rst", "w") as file:
        print(index_rst, file=file)

    def dfs(node):
        for h, v in node.items():
            mod_name = h
            sub_mod_names = list(v.keys())

            with open(h + ".rst", "w") as file:
                rst = save_rst(mod_name, sub_mod_names)
                print(rst, file=file)

            dfs(v)

    dfs(hierarchy)


if __name__ == "__main__":
    print_include_modules()
