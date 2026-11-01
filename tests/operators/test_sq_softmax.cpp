#include <cmath>
#include <cstring>
#include <iostream>

#include "pTensor.h"
#include "gtest/gtest.h"

#include "constants_sq_softmax.hpp"
using std::cout;
using std::endl;

using namespace pTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__0_f) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_0_f);
  Tensor in = new RomTensor({ 64 }, flt, s_ref_in_0_f);
  Tensor out = new RamTensor({ 64 }, flt);

  pTensor::ReferenceOperators::SoftmaxOperator<float> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__0_q) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_0_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_0_f_zp, s_ref_out_0_f_scale));
  Tensor in = new RomTensor({ 64 }, i8, s_ref_in_0_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_0_f_zp, s_ref_in_0_f_scale));
  Tensor out = new RamTensor({ 64 }, flt);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_0_q_zp, s_out_0_q_scale));

  pTensor::ReferenceOperators::SoftmaxOperator<int8_t> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__1_f) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 64 }, flt, s_ref_in_1_f);
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_1_f);
  Tensor out = new RamTensor({ 64 }, flt);

  pTensor::ReferenceOperators::SoftmaxOperator<float> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__1_q) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 64 }, i8, s_ref_in_1_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_1_f_zp, s_ref_in_1_f_scale));
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_1_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_1_f_zp, s_ref_out_1_f_scale));
  Tensor out = new RamTensor({ 64 }, flt);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_1_q_zp, s_out_1_q_scale));

  pTensor::ReferenceOperators::SoftmaxOperator<int8_t> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__2_f) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 64 }, flt, s_ref_in_2_f);
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_2_f);
  Tensor out = new RamTensor({ 64 }, flt);

  pTensor::ReferenceOperators::SoftmaxOperator<float> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__2_q) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 64 }, i8, s_ref_in_2_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_2_f_zp, s_ref_in_2_f_scale));
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_2_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_2_f_zp, s_ref_out_2_f_scale));
  Tensor out = new RamTensor({ 64 }, flt);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_2_q_zp, s_out_2_q_scale));

  pTensor::ReferenceOperators::SoftmaxOperator<int8_t> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__3_f) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 64 }, flt);
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_3_f);
  Tensor in = new RomTensor({ 64 }, flt, s_ref_in_3_f);

  pTensor::ReferenceOperators::SoftmaxOperator<float> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Softmax, random_gen_softmax__3_q) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 64 }, flt);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_3_q_zp, s_out_3_q_scale));
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_3_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_3_f_zp, s_ref_out_3_f_scale));
  Tensor in = new RomTensor({ 64 }, i8, s_ref_in_3_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_3_f_zp, s_ref_in_3_f_scale));

  pTensor::ReferenceOperators::SoftmaxOperator<int8_t> softmaxOp;
  softmaxOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 2);
}
}


/***************************************
 * Regression: a confident prediction (one logit far above the rest) used to
 * underflow every exponent, divide by zero and return 127 for every class.
 ***************************************/

TEST(Softmax, int8_peaked_logits) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<256, uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  // Logits 12.0, -6.0, ..., stored as int8 with scale 0.12, zero point 0.
  static const int8_t logits[10] = {100, -50, -50, -40, -50, -50, -45, -50, -50, -50};
  Tensor in = new RomTensor({1, 10}, i8, logits);
  in->set_quantization_params(PerTensorQuantizationParams(0, 0.12f));
  int8_t probs[10];
  Tensor out = new BufferTensor({1, 10}, i8, probs);
  out->set_quantization_params(PerTensorQuantizationParams(-128, 1.0f / 256.0f));

  pTensor::ReferenceOperators::SoftmaxOperator<int8_t> op;
  op.set_inputs({{pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::in, in}})
      .set_outputs({{pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::out, out}})
      .eval();
  ASSERT_EQ(got_error, false);
  EXPECT_EQ(probs[0], 127);  // p ~= 1.0 saturates at the top of the range
  for (int i = 1; i < 10; i++) EXPECT_EQ(probs[i], -128);
}

TEST(Softmax, int8_matches_float_softmax) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<256, uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  static const int8_t logits[6] = {-128, -60, 0, 17, 40, 127};
  const float scale = 0.05f;
  const int32_t zp = -3;
  Tensor in = new RomTensor({1, 6}, i8, logits);
  in->set_quantization_params(PerTensorQuantizationParams(zp, scale));
  int8_t probs[6];
  Tensor out = new BufferTensor({1, 6}, i8, probs);
  out->set_quantization_params(PerTensorQuantizationParams(-128, 1.0f / 256.0f));
  pTensor::ReferenceOperators::SoftmaxOperator<int8_t> op;
  op.set_inputs({{pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::in, in}})
      .set_outputs({{pTensor::ReferenceOperators::SoftmaxOperator<int8_t>::out, out}})
      .eval();
  ASSERT_EQ(got_error, false);

  float x[6], mx = -1e30f, sum = 0;
  for (int i = 0; i < 6; i++) {
    x[i] = scale * (logits[i] - zp);
    mx = x[i] > mx ? x[i] : mx;
  }
  for (int i = 0; i < 6; i++) sum += std::exp(x[i] - mx);
  for (int i = 0; i < 6; i++) {
    const long q = std::lround(std::exp(x[i] - mx) / sum * 256.0f) - 128;
    const int expected = q < -128 ? -128 : (q > 127 ? 127 : static_cast<int>(q));
    EXPECT_EQ(probs[i], expected) << "class " << i;
  }
}
