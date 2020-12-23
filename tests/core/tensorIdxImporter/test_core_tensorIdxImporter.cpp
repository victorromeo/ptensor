#include "test_helper.h"
#include "src/pTensor/loaders/tensorIdxImporter.hpp"

#include <iostream>
using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;

void test_core_ntoh32(void) {
  
  uint32_t input = 63;
  
  uint32_t result = ntoh32(input);
  
  EXPECT_EQ(result, 1056964608);
}

void test_core_importUchar(void) {
  
  TensorIdxImporter t_import;
  
  Tensor* t =
      t_import.ubyte_import("/fs/constants/idxImport/uint8_4d_power2.idx");
  
  double result = sum<unsigned char>(t);
  EXPECT_EQ(result, 4518);
  delete t;
}

void test_core_importShort(void) {
  
  TensorIdxImporter t_import;
  
  Tensor* t =
      t_import.short_import("/fs/constants/idxImport/int16_4d_power2.idx");
  
  double result = sum<short>(t);
  EXPECT_EQ(result, 270250);
  delete t;
}

void test_core_importInt(void) {
  
  TensorIdxImporter t_import;
  
  Tensor* t =
      t_import.int_import("/fs/constants/idxImport/int32_4d_power2.idx");
  
  double result = sum<int>(t);
  EXPECT_EQ((int)result, 7158278745);
  delete t;
}

void test_core_importFloat(void) {
  
  TensorIdxImporter t_import;
  
  Tensor* t =
      t_import.float_import("/fs/constants/idxImport/float_4d_power2.idx");
  

  double result = sum<float>(t);

  DEBUG("***floating point test yielded: %.8e\r\n", (float)result);
  EXPECT_EQ((float)result, -1.0f);
  delete t;
}

// First configure the pTensor test runner
PTENSOR_TEST_CONFIGURE()

// Second declare tests to run
PTENSOR_TEST(core, ntoh32, "test ntoh32")
PTENSOR_TEST(core, importUchar, "Test unsigned char import")
PTENSOR_TEST(core, importShort, "Test short inport")
//PTENSOR_TEST(core, importInt, "Test import Int")
PTENSOR_TEST(core, importFloat, "Test import float")


// Third, run like hell
PTENSOR_TEST_RUN()

