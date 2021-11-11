# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import pkgutil
import importlib
import inspect
from inspect import ismodule, isclass, isfunction, signature
import re
import toml
import os
import errno
from typing import Any, Tuple

from docs import sparse_module_hierarchy, walk_packages

# 1. Load and validate configuration file
config = toml.load("../../website/documentation.toml")
#print(config)

# Validate module name to document
assert 'settings' in config and 'search' in config['settings'] and (type(config['settings']['search']) is str or type(config['settings']['search']) is list)

# TODO: Try to import module, more validation, etc.

# 2. Construct regular expressions for includes and excludes
# Default include/exclude rules
patterns = {
    'include': {
        'modules': re.compile(r".+"),
        'symbols': re.compile(r".+")
    },
    'exclude': {
        'modules': re.compile(r""),
        'symbols': re.compile(r"")
    }
}

# Override rules based on configuration file
if 'filters' in config:
    filters = config['filters']
    for clude, rules in filters.items():
        for rule, pattern in rules.items():
            if type(pattern) is list:
                pattern = '|'.join(pattern)
            patterns[clude][rule] = re.compile(pattern)


# 3. Read in all modules and symbols
search = config['settings']['search']
search = [search] if type(search) is str else search
modules_and_symbols = {}
for modname in set(search):
    modules_and_symbols = {**modules_and_symbols, **walk_packages(modname)} 

# 4. Apply filtering
# TODO: Would be slightly faster if we applied module filtering inside walk_packages
tmp = {}
for x, y in modules_and_symbols.items():
    if patterns['include']['modules'].fullmatch(x) is not None and patterns['exclude']['modules'].fullmatch(x) is None:

        new_y1 = [(a, b) for a, b in y[1]
            if patterns['include']['symbols'].fullmatch(x+'.'+a) is not None and patterns['exclude']['symbols'].fullmatch(x+'.'+a) is None
        ]

        tmp[x] = (y[0], new_y1)

modules_and_symbols = tmp

# 6. Generate sidebar
# Build hierarchy of modules
hierarchy = sparse_module_hierarchy(modules_and_symbols.keys())

# DEBUG
#for h, v in hierarchy.items():
#    print(h, v)

# Generate index.rst
# Flatten hierarchy
modules = []
def dfs(node):
    for h, v in node.items():
        modules.append(h)   # .replace('_', '\_')
        dfs(v)
dfs(hierarchy)

#modules = ['beanmachine', 'beanmachine.applications']
modules = '\n   '.join(modules)

index_rst = f""".. BeanMachine documentation master file, created by
   sphinx-quickstart on Thu Jun 10 14:00:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

with open('index.rst', 'w') as file:
    print(index_rst, file=file)

# Generate 1 rst for each module
def make_rst(mod_name, sub_mod_names):
    rst = f"""{mod_name}
{"="*len(mod_name)}
"""

    if len(sub_mod_names):
        sub_mods = '\n   '.join(sub_mod_names)
        rst = rst + f"""
Subpackages
-----------

.. toctree::
   :maxdepth: 4

   {sub_mods}
"""

    rst = rst + f"""
Module contents
---------------

.. automodule:: {mod_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    return rst

def dfs(node):
    for h, v in node.items():
        mod_name = h #.replace('_', '\_')
        sub_mod_names = [m for m in v.keys()] # .replace('_', '\_')

        with open(h + '.rst', 'w') as file:
            rst = make_rst(mod_name, sub_mod_names)
            print(rst, file=file)

        #print('mod', mod_name, 'submods', sub_mod_names)

        dfs(v)

dfs(hierarchy)
