# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import re
import sys
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6


TEST_REQUIRES = ["pytest", "pytest-cov"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black==20.8b1",
    "flake8",
    "flake8-bugbear",
    "sphinx",
    "sphinx-autodoc-typehints",
    "usort",
]
TUTORIALS_REQUIRES = ["jupyter", "matplotlib", "cma", "torchvision"]
CPP_COMPILE_ARGS = ["-std=c++14", "-Werror"]


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
    conda_include_dir = os.path.join(os.environ["CONDA_PREFIX"], "include")
    INCLUDE_DIRS.extend([conda_include_dir, os.path.join(conda_include_dir, "eigen3")])

if sys.platform.startswith("linux"):
    INCLUDE_DIRS.extend(
        [
            "/usr/include",
            "/usr/include/eigen3",
            "/usr/include/x86_64-linux-gnu",
        ]
    )

setup(
    name="beanmachine",
    version=version,
    description="Probabilistic Programming Language for Bayesian Inference",
    author="Facebook, Inc.",
    license="MIT",
    url="https://beanmachine.org",
    project_urls={
        "Documentation": "https://beanmachine.org",
        "Source": "https://github.com/facebookincubator/BeanMachine",
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
        "torch>=1.8.1",
        "numpy>=1.18.1",
        "pandas>=0.24.2",
        "plotly>=2.2.1",
        "scipy>=0.16",
        "statsmodels>=0.12.0",
        "tqdm>=4.46.0",
        "astor>=0.7.1",
        "black>=19.3b0",
        "gpytorch>=1.3.0",
        "botorch>=0.3.3",
        "xarray>=0.16.0",
        "arviz>=0.11.0",
        "flowtorch>=0.2",
    ],
    packages=find_packages("src/"),
    package_dir={"": "src"},
    ext_modules=[
        Pybind11Extension(
            name="beanmachine.graph",
            sources=sorted(
                set(glob("src/beanmachine/graph/**/*.cpp", recursive=True))
                - set(glob("src/beanmachine/graph/**/*_test.cpp", recursive=True))
            ),
            include_dirs=INCLUDE_DIRS,
            extra_compile_args=CPP_COMPILE_ARGS,
        )
    ],
    cmdclass={"build_ext": build_ext},
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
