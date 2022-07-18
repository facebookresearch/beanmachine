# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import platform
import re
import shutil
import subprocess
import sys
from distutils.command.build import build as _build
from glob import glob

from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 7

INSTALL_REQUIRES = [
    "arviz>=0.11.0",
    "astor>=0.7.1",
    "black==22.3.0",
    "botorch>=0.5.1",
    "gpytorch>=1.3.0",
    "graphviz>=0.17",
    "functorch>=0.2.0",
    "netCDF4<=1.5.8; python_version<'3.8'",
    "numpy>=1.18.1",
    "pandas>=0.24.2",
    "plotly>=2.2.1",
    "scipy>=0.16",
    "statsmodels>=0.12.0",
    "torch>=1.9.0",
    "tqdm>=4.46.0",
    "typing-extensions>=3.10",
    "xarray>=0.16.0",
]
TEST_REQUIRES = ["pytest>=7.0.0", "pytest-cov"]
TUTORIALS_REQUIRES = [
    "bokeh",
    "cma",
    "ipywidgets",
    "jupyter",
    "lxml>=4.9",
    "matplotlib",
    "mdformat",
    "mdformat-myst",
    "scikit-learn>=1.0.0",
    "seaborn",
    "tabulate",
    "torchvision",
]
DEV_REQUIRES = (
    TEST_REQUIRES
    + TUTORIALS_REQUIRES
    + [
        "flake8==4.0.1",
        "flake8-bugbear",
        "libcst==0.4.1",
        "nbval",
        "sphinx==4.2.0",
        "sphinx-autodoc-typehints",
        "sphinx_rtd_theme",
        "toml>=0.10.2",
        # `black` is included in `INSTALL_REQUIRES` above.
        "ufmt==1.3.2",
        "usort==1.0.2",
    ]
)

if platform.system() == "Windows":
    CPP_COMPILE_ARGS = ["/WX", "/permissive-", "/std:c++20"]
else:
    CPP_COMPILE_ARGS = ["-std=c++2a", "-Werror"]

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
    INCLUDE_DIRS.extend([conda_include_dir, os.path.join(conda_include_dir, "eigen3")])
    INCLUDE_DIRS.extend([conda_include_dir, os.path.join(conda_include_dir, "boost")])

if sys.platform.startswith("linux"):
    INCLUDE_DIRS.extend(
        [
            "/usr/include",
            "/usr/include/eigen3",
            "/usr/include/boost169/",
            "/usr/include/x86_64-linux-gnu",
        ]
    )
elif sys.platform.startswith("darwin"):
    # MacOS dependencies installed through HomeBrew
    INCLUDE_DIRS.extend(
        glob("/usr/local/Cellar/eigen/*/include/eigen3")
        + glob("/usr/local/Cellar/boost/*/include")
    )


# From torch.mlir: "Build phase discovery is unreliable. Just tell it what phases to run."
class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class NoopBuildExtension(build_ext):
    def build_extension(self, ext):
        pass


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_py):
    def build_llvm(self, cmake_build_dir: str):
        if not os.path.isdir(cmake_build_dir):
            c_compiler = os.getenv("C_COMPILER")
            cxx_compiler = os.getenv("CXX_COMPILER")
            print("building LLVM...")
            os.makedirs(cmake_build_dir, exist_ok=True)
            src_dir = os.path.abspath(os.path.dirname(__file__))
            llvm_dir = os.path.join(src_dir, "externals", "llvm-project", "llvm")
            cmake_args = [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLVM_TARGETS_TO_BUILD=Native",
                "-DLLVM_ENABLE_PROJECTS=mlir;llvm",
                f"-B {cmake_build_dir}",
                "-G Ninja",
            ]
            if platform == "darwin":
                cpp_include_path = os.getenv("CPLUS_INCLUDE_PATH")
                if not self.c_compiler or not self.cxx_compiler or not cpp_include_path:
                    raise NotImplementedError(
                        "If you are running on a mac, please set the environment variables:"
                        " C_COMPILER, CXX_COMPILER, CPLUS_INCLUDE_PATH. For example,"
                        "'/usr/local/compiler/clang+llvm-14.0.0-x86_64-apple-darwin/bin/clang' is a valid path to a c compiler"
                        " and `/usr/local/compiler/clang+llvm-14.0.0-x86_64-apple-darwin/include/c++/v1:/Library/Developer/CommandLineTools/SDKs/MacOSX12.3.sdk/usr/include`"
                        " is a valid path for CPLUS_INCLUDE_PATH"
                    )
            if c_compiler:
                cmake_args.append(f"-DCMAKE_C_COMPILER={c_compiler}")
            if cxx_compiler:
                cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx_compiler}")

            subprocess.check_call(["cmake", llvm_dir] + cmake_args, cwd=cmake_build_dir)
            subprocess.check_call(
                ["cmake", "--build", ".", "--target", "check-mlir"], cwd=cmake_build_dir
            )
        else:
            print("skipping LLVM build")

    def build_paic2(self, paic2_src: str, paic2_build_dir: str, cmake_module_path: str):
        c_compiler = os.getenv("C_COMPILER")
        cxx_compiler = os.getenv("CXX_COMPILER")
        os.makedirs(paic2_build_dir, exist_ok=True)
        paic2_cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_MODULE_PATH={cmake_module_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-B {paic2_build_dir}",
            "-G Ninja",
        ]
        if c_compiler:
            paic2_cmake_args.append(f"-DCMAKE_C_COMPILER={c_compiler}")
        if cxx_compiler:
            paic2_cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx_compiler}")
        subprocess.check_call(
            ["cmake", paic2_src] + paic2_cmake_args, cwd=paic2_build_dir
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=paic2_build_dir)

    def run(self):
        src_dir = os.path.abspath(os.path.dirname(__file__))
        print("starting cmake build...")

        # build llvm
        cmake_build_dir = os.path.join(src_dir, "build")
        self.build_llvm(cmake_build_dir)

        # build PAIC2
        paic2_dir = os.path.join(src_dir, "src", "beanmachine", "paic2")
        paic2_build_dir = os.path.join(paic2_dir, "build")
        self.build_paic2(paic2_dir, paic2_build_dir, f"{cmake_build_dir}")

        # copy paic2 modules into src
        target_dir = self.build_lib
        regex = re.compile("(.*so$)")
        for root, dirs, files in os.walk(paic2_build_dir):
            for file in files:
                print(file.__str__())
                if regex.match(file):
                    dest = os.path.join(target_dir, os.path.basename(file))
                    print("copying " + file.__str__() + " into " + dest.__str__())
                    full_path = os.path.join(paic2_build_dir, file)
                    shutil.copyfile(full_path, dest)


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
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"beanmachine/ppl": ["py.typed"]},
    ext_modules=[
        Pybind11Extension(
            name="beanmachine.graph",
            sources=sorted(
                set(glob("src/beanmachine/graph/**/*.cpp", recursive=True))
                - set(glob("src/beanmachine/graph/**/*_test.cpp", recursive=True))
            ),
            include_dirs=INCLUDE_DIRS,
            extra_compile_args=CPP_COMPILE_ARGS,
        ),
        CMakeExtension(name="paic2"),
    ],
    cmdclass={"build": CustomBuild, "build_py": CMakeBuild, "build_ext": build_ext},
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "tutorials": TUTORIALS_REQUIRES,
    },
)
