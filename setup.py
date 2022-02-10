# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import re
import sys
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 7


TEST_REQUIRES = ["pytest", "pytest-cov"]
TUTORIALS_REQUIRES = [
    "bokeh",
    "cma",
    "ipywidgets",
    "jupyter",
    "matplotlib",
    "scikit-learn>=1.0.0",
    "seaborn",
    "torchvision",
]
DEV_REQUIRES = (
    TEST_REQUIRES
    + TUTORIALS_REQUIRES
    + [
        "black==20.8b1",
        "flake8",
        "flake8-bugbear",
        "sphinx==4.2.0",
        "sphinx-autodoc-typehints",
        "sphinx_rtd_theme",
        "toml>=0.10.2",
        "usort",
        "nbval",
    ]
)

CPP_COMPILE_ARGS = ["-fPIC"]
if platform.system() == "Windows":
    CPP_COMPILE_ARGS = ["/WX", "/permissive-", "-DEIGEN_HAS_C99_MATH"]


# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


# get version string from module
current_dir = os.path.dirname(os.path.abspath(__file__))
init_file = os.path.join(current_dir, "src", "beanmachine", "__init__.py")
version_regexp = r"__version__ = ['\"]([^'\"]*)['\"]"
with open(init_file, "r") as f:
    version = re.search(version_regexp, f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Use absolute path to the src directory
INCLUDE_DIRS = [os.path.join(current_dir, "src")]

# check if we're installing in a conda environment
if "CONDA_PREFIX" in os.environ:
    conda_inc = "Library/include" if platform.system() == "Windows" else "include"
    conda_include_dir = os.path.join(os.environ["CONDA_PREFIX"], conda_inc)
    INCLUDE_DIRS.extend(
        [
            conda_include_dir,
            # os.path.join(conda_include_dir, "eigen3"),
            "/root/micromamba/lib/python3.9/site-packages/torch/include/torch/csrc/api/include",
            "/root/micromamba/lib/python3.9/site-packages/torch/include",
            # "/workspace/libtorch/include",
            # "/workspace/libtorch/include/torch/csrc/api/include",
        ]
    )  # TODO: replace with aten
    INCLUDE_DIRS.extend([conda_include_dir, os.path.join(conda_include_dir, "boost")])

if sys.platform.startswith("linux"):
    INCLUDE_DIRS.extend(
        [
            "/usr/include",
            # "/usr/include/eigen3",  # TODO: replace with aten
            "/usr/include/boost169/",
            "/usr/include/x86_64-linux-gnu",
        ]
    )

setup(
    name="beanmachine",
    version=version,
    description="Probabilistic Programming Language for Bayesian Inference",
    author="Meta Platforms, Inc.",
    license="MIT",
    url="https://beanmachine.org",
    project_urls={
        "Documentation": "https://beanmachine.org",
        "Source": "https://github.com/facebookresearch/beanmachine",
    },
    keywords=[
        "Probabilistic Programming Language",
        "Bayesian Inference",
        "Statistical Modeling",
        "MCMC",
        "Variational Inference",
        "PyTorch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">={}.{}".format(REQUIRED_MAJOR, REQUIRED_MINOR),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.18.1",
        "pandas>=0.24.2",
        "plotly>=2.2.1",
        "scipy>=0.16",
        "statsmodels>=0.12.0",
        "tqdm>=4.46.0",
        "astor>=0.7.1",
        "black>=19.3b0",
        "gpytorch>=1.3.0",
        "botorch>=0.5.1",
        "xarray>=0.16.0",
        "arviz>=0.11.0",
        "flowtorch>=0.3",
        "parameterized>=0.8.1",
        "graphviz>=0.17",
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        Pybind11Extension(
            name="beanmachine.graph",
            sources=sorted(
                (
                    set([
                        "src/beanmachine/graph/pybindings.cpp",
                        "src/beanmachine/graph/graph.cpp",
                        "src/beanmachine/graph/profiler.cpp",
                        "src/beanmachine/graph/to_dot.cpp",
                        "src/beanmachine/graph/perf_report.cpp",
                        "src/beanmachine/graph/support.cpp",
                        "src/beanmachine/graph/util.cpp",
                        "src/beanmachine/graph/distribution/distribution.cpp",
                    ])
                    | set(glob("src/beanmachine/graph/transform/*.cpp", recursive=True))
                    | set(glob("src/beanmachine/graph/factor/*.cpp", recursive=True))
                )
                # set(glob("src/beanmachine/graph/**/*.cpp", recursive=True))
                - set([
                    "src/beanmachine/graph/operator/gradient.cpp",
                    "src/beanmachine/graph/operator/linalgop.cpp",
                    "src/beanmachine/graph/operator/multiaryop.cpp",
                    "src/beanmachine/graph/operator/unaryop.cpp",
                ])
                - set(glob("src/beanmachine/graph/**/*_test.cpp", recursive=True))
            ),
            include_dirs=INCLUDE_DIRS,
            # extra_compile_args=CPP_COMPILE_ARGS,
            extra_compile_args=[
                '-DUSE_C10D_GLOO', 
                '-DUSE_DISTRIBUTED', 
                '-DUSE_RPC', 
                '-DUSE_TENSORPIPE', 
                '-isystem /root/micromamba/lib/python3.9/site-packages/torch/include',
                '-isystem /root/micromamba/lib/python3.9/site-packages/torch/include/torch/csrc/api/include',
                '-D_GLIBCXX_USE_CXX11_ABI=0',
                '-D_GLIBCXX_USE_CXX11_ABI=0'
            ],
            extra_link_args=[
                '-D_GLIBCXX_USE_CXX11_ABI=0',
                '-rdynamic',
                '-Wl,-rpath,/root/micromamba/lib/python3.9/site-packages/torch/lib /root/micromamba/lib/python3.9/site-packages/torch/lib/libtorch.so /root/micromamba/lib/python3.9/site-packages/torch/lib/libc10.so',
                '-Wl,--no-as-needed,/root/micromamba/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so',
                '-Wl,--as-needed,/root/micromamba/lib/python3.9/site-packages/torch/lib/libc10.so',
                '-lpthread',
                '-Wl,--no-as-needed,/root/micromamba/lib/python3.9/site-packages/torch/lib/libtorch.so',
                '-Wl,--as-needed',
            ],
            cxx_std=14,
        )
    ],
    cmdclass={"build_ext": build_ext},
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
