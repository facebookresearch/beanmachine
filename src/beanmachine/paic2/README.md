Paic2 Installation

Paic2 installation has been tested on macos 12.3.1. 
I have been using LLVM Clang version 14 to compile.

1. Install Conda, Mamba, Ninja, cmake, and LLVM version 14. LLVM artifacts can be found here: https://github.com/llvm/llvm-project/releases
2. Navigate to root of beanmachine
3. `mamba env create -f src/beanmachine/paic2/environment.yml`
4. `mamba activate bean-machine-39`
5. `pip install -v -e .[dev]`
6. `export C_COMPILER={path to llvm clang}`
7. `export CXX_COMPILER={path to llvm clang++}`
8. If using a Mac, set the environment variable CPLUS_INCLUDE_PATH. For example, if you installed LLVM to /usr/local/compiler, you may set the environment variable to something like this: `export CPLUS_INCLUDE_PATH=/usr/local/compiler/clang+llvm-14.0.0-x86_64-apple-darwin/include/c++/v1:/Library/Developer/CommandLineTools/SDKs/MacOSX12.3.sdk/usr/include`
9. `python src/beanmachine/paic2/setup_paic2.py $(pwd)` Note that the first time you run this the build will take a long time because it is building llvm from source.
10. `PYTHONPATH=$(pwd)/src pytest -v -s $(pwd)/src/beanmachine/paic2/test/build_test.py`

If the installation was successful, the output should be the following:

```
src/beanmachine/paic2/test/build_test.py::BuildTest::test_paic2_is_imported module {
  func.func @foo() {
    return
  }
}
PASSED
```