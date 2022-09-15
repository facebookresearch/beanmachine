/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <string>
#include "beanmachine/graph/profiler.h"

namespace beanmachine {
namespace graph {

TEST(profilertest, output) {
  Profiler* profiler = new (Profiler);
  std::string expected[12];
  expected[0] = "Title: Bean Machine Graph Profiler Report";
  expected[1] = "Generated at:";
  expected[2] = "Checking integrity of profiled event list";
  expected[3] = "No mismatched events";
  expected[4] = "Integrity check complete. No errors";
  expected[5] = "List of profiled events:";
  expected[6] = "   Test 0:[1]";
  expected[7] = "      Test 1:[1]";
  expected[8] = "         Test 2:[1]";
  expected[9] = "            Test 3:[1]";
  expected[10] = "            Test 4:[1]";
  expected[11] = "         Test 5:[1]";

  profiler->activate();

  profiler->start("Test 0");
  profiler->start("Test 1");
  profiler->start("Test 2");
  profiler->start("Test 3");
  profiler->stop("Test 3");
  profiler->start("Test 4");
  profiler->stop("Test 4");
  profiler->stop("Test 2");
  profiler->start("Test 5");
  profiler->stop("Test 5");
  profiler->stop("Test 1");
  profiler->stop("Test 0");
  std::string filename = "/tmp/new_profiler_test" + std::to_string(getpid());

  remove(filename.c_str());
  profiler->get_report(filename);
  std::ifstream fin(filename.c_str());
  for (int i = 0; i < 1; i++) {
    char s[100];
    for (int j = 0; j < 100; j++) {
      s[j] = '\0';
    }
    fin.getline(s, 90);
    std::string ss(s);
    EXPECT_EQ(ss, expected[i]);
  }
  for (int i = 1; i < 2; i++) {
    char s[100];
    for (int j = 0; j < 100; j++) {
      s[j] = '\0';
    }
    fin.getline(s, 90, ':');
    fin.ignore(90, '\n');
    std::string ss(s);
    EXPECT_EQ(ss + ":", expected[i]);
  }
  for (int i = 2; i < 5; i++) {
    char s[100];
    for (int j = 0; j < 100; j++) {
      s[j] = '\0';
    }
    fin.getline(s, 90);
    std::string ss(s);
    EXPECT_EQ(ss, expected[i]);
  }
  for (int i = 5; i < 6; i++) {
    char s[100];
    for (int j = 0; j < 100; j++) {
      s[j] = '\0';
    }
    fin.getline(s, 90, ':');
    fin.ignore(90, '\n');
    std::string ss(s);
    EXPECT_EQ(ss + ":", expected[i]);
  }
  for (int i = 6; i < 12; i++) {
    char s[100];
    for (int j = 0; j < 100; j++) {
      s[j] = '\0';
    }
    fin.getline(s, 90, ']');
    fin.ignore(90, '\n');
    std::string ss(s);
    EXPECT_EQ(ss + "]", expected[i]);
  }

  remove(filename.c_str());
  delete profiler;
}

TEST(profilertest, BadStop) {
  Profiler* p = new (Profiler);
  p->activate();
  // Stopping what has not started
  std::string event_name("Test 2");
  EXPECT_THROW(p->stop(event_name), std::runtime_error);
  try {
    p->stop(event_name);
    ADD_FAILURE();
  } catch (std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), p->badstop_exception + event_name);
    SUCCEED();
  }
  delete p;
}

TEST(profilertest, StartWithoutStop) {
  Profiler* p = new (Profiler);
  p->activate();
  // Starting test 1 and not stopping it
  std::string event_name("Test 0");
  p->start(event_name);
  EXPECT_THROW(p->get_report("/dev/null"), std::runtime_error);
  delete p;

  p = new (Profiler);
  try {
    p->activate();
    p->start(event_name);
    p->get_report("/dev/null");
    ADD_FAILURE();
  } catch (std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), p->failed_integrity);
    SUCCEED();
  }
  delete p;
}

TEST(profilertest, MoreStopsThanStarts) {
  Profiler* p = new (Profiler);
  p->activate();
  // Stopping an event after it has stopped!
  std::string event_name("Test 0");
  try {
    p->start(event_name);
    p->stop(event_name);
    p->stop(event_name);
    p->get_report("/dev/null");
    ADD_FAILURE();
  } catch (std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), p->failed_integrity);
    SUCCEED();
  }
  delete p;
}

TEST(profilertest, NoEvents) {
  Profiler* p = new (Profiler);
  p->activate();
  EXPECT_NO_THROW(p->get_report("/dev/null"));
  delete p;
}

TEST(profilertest, BadFile) {
  Profiler* p = new (Profiler);
  p->activate();
  std::string badfilename("/NonExisting/NonExisting/stat.txt");
  try {
    p->get_report(badfilename);
    ADD_FAILURE();
  } catch (std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), p->badfile_exception);
    SUCCEED();
  }
  delete p;
}

TEST(profilertest, UnnestedEvents) {
  Profiler* p = new (Profiler);
  p->activate();

  std::string event0("Test 0");
  std::string event1("Test 1");

  p->start(event0);
  p->start(event1);
  // stopping event0 before event1 !!! User error
  p->stop(event0);
  p->stop(event1);
  try {
    p->get_report("/dev/null");
    ADD_FAILURE();
  } catch (std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), p->notnested_exception);
    SUCCEED();
  }
  delete p;
}

} // namespace graph
} // namespace beanmachine
