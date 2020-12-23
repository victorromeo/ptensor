#include "test_helper.h"

// Default to using GTest like asserts and expects as these give more info that unity
// We will forward these commands to unity in test_helper.h
void test_sanity_checkTrue(){
    EXPECT_EQ(true, true);
}

void test_sanity_checkEqual(){
    EXPECT_EQ(1 == 1, true);
}

// First configure the pTensor test runner
PTENSOR_TEST_CONFIGURE()

// Second declare tests to run
PTENSOR_TEST(sanity, checkTrue, "Sanity check number 1")
PTENSOR_TEST(sanity, checkEqual, "Sanity check number 2")


// Third, run like hell
PTENSOR_TEST_RUN()
