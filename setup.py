# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import re
import sys
from glob import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6


TEST_REQUIRES = ["pytest", "pytest-cov", "gpytorch"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black==19.3b0",
    "isort",
    "flake8",
    "sphinx",
    "sphinx-autodoc-typehints",
]
TUTORIALS_REQUIRES = ["jupyter", "matplotlib", "cma", "torchvision"]


EXTRA_COMPILE_ARGS = ["-std=c++14", "-Werror"]


BOOST_HEADER = "/boost/math/special_functions/polygamma.hpp"


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
current_dir = os.path.dirname(__file__)
init_file = os.path.join(current_dir, "beanmachine", "__init__.py")
version_regexp = r"__version__ = ['\"]([^'\"]*)['\"]"
with open(init_file, "r") as f:
    version = re.search(version_regexp, f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()


def find_include_dirs():
    inc_dir = {os.getcwd()}
    if "--boost_include_dir" in sys.argv:
        index = sys.argv.index("--boost_include_dir")
        sys.argv.pop(index)
        dir_candidate = sys.argv.pop(index)
        header_dir = dir_candidate + BOOST_HEADER
        if os.path.isfile(header_dir):
            inc_dir.add(dir_candidate)
            return list(inc_dir)
        raise FileNotFoundError("Boost is not found in the given directory.")
    if sys.platform == "win32":
        raise ValueError("Please provide a valid --boost_include_dir.")
    locs = os.popen("whereis boost").read().split(" ")
    for loc in locs:
        dir_candidate = os.path.dirname(loc)
        header_dir = dir_candidate + BOOST_HEADER
        if os.path.isfile(header_dir):
            inc_dir.add(dir_candidate)
            return list(inc_dir)
    raise FileNotFoundError(
        "Boost is not found. Please install Boost first,"
        + " then provide a valid --boost_include_dir."
    )


bmg_inc_dir = find_include_dirs()

setup(
    name="BeanMachine",
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
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.4.0",
        "dataclasses>=0.6",
        "pandas>=0.24.2",
        "plotly>=2.2.1",
        "scipy>=0.16",
        "statsmodels>=0.8.0",
        "tqdm>=4.40.2",
        "astor>=0.7.1",
        "black>=19.3b0",
    ],
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="beanmachine.graph",
            sources=list(
                set(glob("beanmachine/graph/**/*.cpp", recursive=True))
                - set(glob("beanmachine/graph/**/*_test.cpp", recursive=True))
            ),
            include_dirs=bmg_inc_dir,
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
        CppExtension(
            name="beanmachine.ppl.utils.tensorops",
            sources=["beanmachine/ppl/utils/tensorops.cpp"],
            include_dirs=[os.getcwd()],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
