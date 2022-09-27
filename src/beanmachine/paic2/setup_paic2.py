import os
import platform
import re
import shutil
import subprocess
import sys
from typing import List


def cmake_build_release(
    src_dir: str,
    build_dir: str,
    cmake_args: List[str],
    build_args: List[str],
    name: str,
):
    if os.path.isdir(build_dir):
        print(
            f"Skipping {name} build. Delete build directory if you want to rebuild it."
        )
        return
    os.makedirs(build_dir, exist_ok=True)
    cmake_args.append(f"-B {build_dir}")
    cmake_args.append("-G Ninja")
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    c_compiler = os.getenv("C_COMPILER")
    cxx_compiler = os.getenv("CXX_COMPILER")
    if c_compiler:
        cmake_args.append(f"-DCMAKE_C_COMPILER={c_compiler}")
    if cxx_compiler:
        cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx_compiler}")
    cmake_result = subprocess.run(
        ["cmake", src_dir] + cmake_args, cwd=build_dir, check=False, capture_output=True
    )
    if cmake_result.returncode != 0:
        print(str(cmake_result.stdout))
        print(str(cmake_result.stderr))
    build_result = subprocess.run(
        build_args, cwd=build_dir, check=True, capture_output=True
    )
    if build_result.returncode != 0:
        print(str(build_result.stdout))
        print(str(build_result.stderr))


def build_llvm(src_dir: str, cmake_build_dir: str):
    if not os.path.isdir(cmake_build_dir):
        c_compiler = os.getenv("C_COMPILER")
        cxx_compiler = os.getenv("CXX_COMPILER")
        print("building LLVM...")
        os.makedirs(cmake_build_dir, exist_ok=True)
        llvm_dir = os.path.join(src_dir, "externals", "llvm-project", "llvm")
        cmake_args = [
            "-DLLVM_TARGETS_TO_BUILD=Native",
            "-DLLVM_ENABLE_PROJECTS=mlir;llvm",
            "-DLLVM_REQUIRES_RTTI=ON",
        ]
        if platform.system() == "Darwin":
            cpp_include_path = os.getenv("CPLUS_INCLUDE_PATH")
            if not c_compiler or not cxx_compiler or not cpp_include_path:
                raise NotImplementedError(
                    "If you are running on a mac, please set the environment variables:"
                    " C_COMPILER, CXX_COMPILER, CPLUS_INCLUDE_PATH. For example,"
                    "'/usr/local/compiler/clang+llvm-14.0.0-x86_64-apple-darwin/bin/clang' is a valid path to a c compiler"
                    " and `/usr/local/compiler/clang+llvm-14.0.0-x86_64-apple-darwin/include/c++/v1:/Library/Developer/CommandLineTools/SDKs/MacOSX12.3.sdk/usr/include`"
                    " is a valid path for CPLUS_INCLUDE_PATH"
                )
        cmake_build_release(
            llvm_dir,
            cmake_build_dir,
            cmake_args,
            ["cmake", "--build", ".", "--target", "check-mlir"],
            "llvm",
        )


def build_pybind11(base_dir: str):
    pybind11_dir = os.path.join(
        base_dir, "src", "beanmachine", "paic2", "externals", "pybind11"
    )
    build_dir = os.path.join(pybind11_dir, "build")
    cmake_args = [f"-DPYTHON_EXECUTABLE={sys.executable}"]
    cmake_build_release(
        pybind11_dir, build_dir, cmake_args, ["cmake", "--build", "."], "pybind11"
    )


def build_paic2(
    base_dir: str, paic2_src: str, paic2_build_dir: str, cmake_module_path: str
):
    mlir_dir = os.path.join(cmake_module_path, "lib", "cmake", "mlir")
    llvm_dir = os.path.join(cmake_module_path, "lib", "cmake", "llvm")
    paic2_cmake_args = [
        f"-DCMAKE_MODULE_PATH={cmake_module_path}",
        f"-DMLIR_DIR={mlir_dir}",
        f"-DLLVM_DIR={llvm_dir}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DBM_ROOT={base_dir}",
    ]
    cmake_build_release(
        paic2_src, paic2_build_dir, paic2_cmake_args, ["cmake", "--build", "."], "paic2"
    )


# Given: target directory for the paic2.so file
def install(base_dir: str, target_dir: str):
    cmake_build_dir = os.path.join(base_dir, "build")
    paic2_dir = os.path.join(base_dir, "src", "beanmachine", "paic2")
    paic2_build_dir = os.path.join(paic2_dir, "build")

    build_llvm(base_dir, cmake_build_dir)
    build_pybind11(base_dir)
    build_paic2(base_dir, paic2_dir, paic2_build_dir, f"{cmake_build_dir}")

    # copy paic2 modules into src
    regex = re.compile("(.*so$)")
    for root, dirs, files in os.walk(paic2_build_dir):
        for file in files:
            if regex.match(file):
                dest = os.path.join(target_dir, os.path.basename(file))
                print("copying " + file.__str__() + " into " + dest.__str__())
                full_path = os.path.join(paic2_build_dir, file)
                shutil.copyfile(full_path, dest)


if __name__ == "__main__":
    src_dir = str(sys.argv[1])
    target_dir = os.path.join(src_dir, "src")
    install(base_dir=src_dir, target_dir=target_dir)
