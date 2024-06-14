cmake_minimum_required(VERSION 3.22)
project(PAIC_IR LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)
find_package(pybind11 REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using pybind11.cmake in: ${pybind11_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(HandleLLVMOptions)

# Python
include(MLIRDetectPythonEnv)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Found python libraries: ${Python3_LIBRARIES}")
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(src)

# pybind module
set(CMAKE_CXX_FLAGS "-g -O0")
pybind11_add_module(paic_mlir src/MLIRBuilder.cpp src/PaicAST.cpp)
target_include_directories(paic_mlir PUBLIC include)
target_link_libraries(paic_mlir
        PRIVATE
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRDialect
        MLIRLLVMCommonConversion
        MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRef
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRPythonSources
        MLIRPythonExtension.Core
        MLIRPythonExtension.AllPassesRegistration
        MLIRPythonExtension.ExecutionEngine
        )
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "dialect_libs: ${dialect_libs}")
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(STATUS "conversion_libs: ${conversion_libs}")
set(LIBS ${dialect_libs} ${conversion_libs})
target_link_libraries(paic_mlir PRIVATE ${LIBS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(paic_mlir
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
