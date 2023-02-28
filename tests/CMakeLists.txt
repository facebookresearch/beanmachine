# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Paths
# ------------------------------------------------------------------------------
file(GLOB_RECURSE bmg_test_src graph/*_test.cpp)

# Targets
# ------------------------------------------------------------------------------
add_executable(bmg_test "")
target_compile_options(bmg_test PRIVATE -O2 -Werror)
target_link_libraries(bmg_test PRIVATE bmg GTest::gtest_main)
target_sources(bmg_test PRIVATE ${bmg_test_src})

enable_testing()
include(GoogleTest)
gtest_discover_tests(bmg_test)
