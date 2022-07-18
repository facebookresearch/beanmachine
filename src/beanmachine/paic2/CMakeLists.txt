cmake_minimum_required(VERSION 3.22)
project(PAIC2 LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

# pybind11
add_subdirectory(${BM_ROOT}/src/beanmachine/paic2/externals/pybind11)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

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
include(MLIRDetectPythonEnv)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# library
add_llvm_library(PAIC2_LIB src/MLIRBuilder.cpp src/PaicAST.cpp)
target_compile_options (PAIC2_LIB PUBLIC -fexceptions)
target_compile_options (PAIC2_LIB PUBLIC -frtti)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs})
target_link_libraries(PAIC2_LIB PRIVATE ${LIBS})

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(PAIC2_LIB
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

# pybind module
set(CMAKE_CXX_FLAGS "-g -O0")
pybind11_add_module(paic2 src/pymod.cpp)
target_include_directories(paic2 PUBLIC include)
target_link_libraries(paic2 PRIVATE PAIC2_LIB)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(paic2
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})