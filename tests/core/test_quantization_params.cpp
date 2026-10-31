#include <iostream>
#include "gtest/gtest.h"
#include "pTensor.h"

using std::cout;
using std::endl;

using namespace pTensor;

SimpleErrorHandler mErrHandler(10);

TEST(QuantizationParams, base_params) {
  bool result = false;
  mErrHandler.set_onError([&result](Error* err){
    if(*err == AttemptToUseUnSpecializedQuantizeParamsError()) {
      result = true; 
    }
  });

  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  QuantizationParams mParams;
  testing::internal::CaptureStdout(); // Dont print error messages since they will be confusing
  int32_t i = mParams.get_zeroP_for_channel(0);
  EXPECT_EQ(result, true);
  EXPECT_EQ(i, 0);
  
  result = false;
  float f = mParams.get_scale_for_channel(0);
  EXPECT_EQ(result, true);
  EXPECT_EQ(f, 0.0f);
  testing::internal::GetCapturedStdout();
  mErrHandler.set_onError([](Error*) {});
}

// Per-tensor params must own their values: passing literals or temporaries
// (or an `int` where int32_t is `long`) used to leave dangling pointers.
static PerTensorQuantizationParams make_params_from_temporaries() {
  return PerTensorQuantizationParams(-7, 0.25f);
}

TEST(QuantizationParams, per_tensor_owns_values) {
  PerTensorQuantizationParams p = make_params_from_temporaries();
  // Clobber the stack region the temporaries lived in.
  volatile int32_t scratch[64];
  for (int i = 0; i < 64; ++i) scratch[i] = 0x5A5A5A5A;
  (void)scratch;
  EXPECT_EQ(p.get_zeroP_for_channel(0), -7);
  EXPECT_FLOAT_EQ(p.get_scale_for_channel(0), 0.25f);
  EXPECT_EQ(p.num_channels(), 1);

  PerTensorQuantizationParams copy(p);
  PerTensorQuantizationParams assigned(0, 1.0f);
  assigned = p;
  EXPECT_EQ(copy.get_zeroP_for_channel(0), -7);
  EXPECT_EQ(assigned.get_zeroP_for_channel(0), -7);
  EXPECT_FLOAT_EQ(assigned.get_scale_for_channel(0), 0.25f);
}
