# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Paths
# ------------------------------------------------------------------------------
file(GLOB_RECURSE bmg_src beanmachine/graph/*.cpp)
list(FILTER bmg_src EXCLUDE REGEX ".*pybindings\.cpp$")

# Targets
# ------------------------------------------------------------------------------
add_library(bmg "")
target_compile_options(bmg PRIVATE -O2 -Werror)
target_link_libraries(bmg PUBLIC ${BOOST_LIBRARY} ${EIGEN3_LIBRARY} range-v3)
target_include_directories(bmg PUBLIC . ${CMAKE_PREFIX_PATH}/include/eigen3)
target_sources(bmg PRIVATE ${bmg_src})
