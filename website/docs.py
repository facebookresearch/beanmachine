# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import types
from collections import OrderedDict
from functools import lru_cache
from inspect import isclass, isfunction, ismodule
from typing import Any, Dict, Sequence, Mapping, Tuple

# Essentially, the list of modules in the navigation sidebar
include_modules = [
    "beanmachine.applications",
    "beanmachine.applications.clara.nmc",
    "beanmachine.applications.clara.nmc_df",
    "beanmachine.ppl",
    "beanmachine.ppl.compiler",
    "beanmachine.ppl.compiler.bmg_types",
	"beanmachine.ppl.compiler.hint",
	"beanmachine.ppl.compiler.patterns",
	"beanmachine.ppl.compiler.performance_report",
	"beanmachine.ppl.compiler.profiler",
	"beanmachine.ppl.compiler.rules",
	"beanmachine.ppl.compiler.single_assignment",
    "beanmachine.ppl.diagnostics",
    "beanmachine.ppl.diagnostics.common_plots",
    "beanmachine.ppl.diagnostics.common_statistics",
    "beanmachine.ppl.diagnostics.diagnostics",
]


def ispublic(name: str) -> bool:
    return not name.startswith("_")


@lru_cache(maxsize=1)
def _module_hierarchy() -> Mapping[str, Any]:
    # Make list of modules to search and their hierarchy
    results: Dict[str, Any] = OrderedDict()
    for module in sorted(include_modules):
        submodules = module.split(".")
        this_dict = results.setdefault(submodules[0], {})

        for idx in range(1, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            this_dict.setdefault(submodule, {})
            this_dict = this_dict[submodule]

    return results


def fullname(o):
    return ".".join([o.__module__, o.__name__])

#@lru_cache(maxsize=1)
def _documentable_modules(module_hierarchy: Mapping[str, Any] = _module_hierarchy()) -> Mapping[types.ModuleType, Sequence[Tuple[str, Any]]]:
    """
    Returns a list of (module, [(name, entity), ...]) pairs for modules
    that are documentable
    """

    # TODO: Self document flowtorch.docs module
    results = {}

    def dfs(dict: Mapping[str, Any]) -> None:
        for key, val in dict.items():
            module = importlib.import_module(key)

            # Get all classes and functions imported/defined in module
            entities = [
                (n, getattr(module, n))
                for n in sorted(
                    [
                        n
                        for n in dir(module)
                        if ispublic(n)
                        and (
                            isclass(getattr(module, n))
                            or isfunction(getattr(module, n))
                        )
                    ]
                )
            ]

            # Filter out ones that aren't defined in this module or a child submodule
            entities = [
                (n, m) for n, m in entities if isfunction(m) or (m.__module__.startswith(module.__name__))
            ]

            results[module] = entities

            dfs(val)

    # Depth first search over module hierarchy, loading modules and extracting entities
    dfs(module_hierarchy)
    return results


def _documentable_entities(documentable_modules: Mapping[types.ModuleType, Sequence[Tuple[str, Any]]] = _documentable_modules()) -> Tuple[Sequence[str], Dict[str, Any]]:
    """
    Returns a list of (str, entity) pairs for entities that are documentable
    """

    name_entity_mapping = {}
    for module, entities in documentable_modules.items():
        if len(entities) > 0:
            name_entity_mapping[module.__name__] = module

        for name, entity in entities:
            qualified_name = f"{module.__name__}.{name}"
            name_entity_mapping[qualified_name] = entity

    sorted_entity_names = sorted(name_entity_mapping.keys())
    return sorted_entity_names, name_entity_mapping


@lru_cache(maxsize=1)
def _sparse_module_hierarchy() -> Mapping[str, Any]:
    # Make list of modules to search and their hierarchy, pruning entries that
    # aren't in include_modules
    results: Dict[str, Any] = OrderedDict()
    this_dict = results

    for module in sorted(include_modules):
        submodules = module.split(".")

        # Navigate to the correct insertion place for this module
        for idx in range(0, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            if submodule in this_dict:
                this_dict = this_dict[submodule]

        # Insert module if it doesn't exist already
        this_dict.setdefault(module, {})

    return results


def generate_markdown(name: str, entity: Any) -> Tuple[str, str]:
    """
    TODO: Method that inputs an object, extracts signature/docstring,
    and formats as markdown
    TODO: Method that build index markdown for overview files
    The overview for the entire API is a special case
    """

    if name == "":
        header = """---
id: overview
sidebar_label: "Overview"
slug: "/api"
---"""
        filename = "../../website/docs/api/overview.mdx"
        return filename, header

    # Regular modules/functions
    item = {
        "id": name,
        "sidebar_label": "Overview" if ismodule(entity) else name.split(".")[-1],
        "slug": f"/api/{name}",
        "ref": entity,
        "filename": f"../../website/docs/api/{name}.mdx",
    }

    header = f"""---
id: {item['id']}
sidebar_label: {item['sidebar_label']}
slug: {item['slug']}
---"""

    markdown = header
    return item["filename"], markdown


module_hierarchy = _module_hierarchy()
documentable_modules = _documentable_modules()

sparse_module_hierarchy = _sparse_module_hierarchy()
sparse_documentable_modules = _documentable_modules(sparse_module_hierarchy)

sorted_entity_names, name_entity_mapping = _documentable_entities()

sparse_sorted_entity_names, sparse_name_entity_mapping = _documentable_entities(sparse_documentable_modules)

__all__ = [
    "documentable_modules",
    "sparse_documentable_modules",
    "generate_markdown",
    "module_hierarchy",
    "sparse_module_hierarchy",
    "name_entity_mapping",
    "sorted_entity_names",
    "sparse_sorted_entity_names",
    "sparse_name_entity_mapping",
]
