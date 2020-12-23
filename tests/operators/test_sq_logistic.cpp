#include <cstring>
#include <iostream>

#include "pTensor.h"
#include "gtest/gtest.h"

#include "constants_sq_logistic.hpp"
using std::cout;
using std::endl;

using namespace pTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(Logistic, random_gen_logistic__0_f) {
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

  Tensor in = new RomTensor({ 64 }, flt, s_ref_in_0_f);
  Tensor out = new RamTensor({ 64 }, flt);
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_0_f);

  pTensor::ReferenceOperators::LogisticOperator<float> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::out, out }
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

TEST(Logistic, random_gen_logistic__0_q) {
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

  Tensor in = new RomTensor({ 64 }, i8, s_ref_in_0_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_0_f_zp, s_ref_in_0_f_scale));
  Tensor out = new RamTensor({ 64 }, flt);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_0_q_zp, s_out_0_q_scale));
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_0_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_0_f_zp, s_ref_out_0_f_scale));

  pTensor::ReferenceOperators::LogisticOperator<int8_t> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::out, out }
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

TEST(Logistic, random_gen_logistic__1_f) {
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

  pTensor::ReferenceOperators::LogisticOperator<float> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::out, out }
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

TEST(Logistic, random_gen_logistic__1_q) {
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

  pTensor::ReferenceOperators::LogisticOperator<int8_t> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::out, out }
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

TEST(Logistic, random_gen_logistic__2_f) {
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

  pTensor::ReferenceOperators::LogisticOperator<float> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::out, out }
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

TEST(Logistic, random_gen_logistic__2_q) {
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

  pTensor::ReferenceOperators::LogisticOperator<int8_t> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::out, out }
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

TEST(Logistic, random_gen_logistic__3_f) {
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

  Tensor in = new RomTensor({ 64 }, flt, s_ref_in_3_f);
  Tensor out = new RamTensor({ 64 }, flt);
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_3_f);

  pTensor::ReferenceOperators::LogisticOperator<float> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<float>::out, out }
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

TEST(Logistic, random_gen_logistic__3_q) {
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

  Tensor in = new RomTensor({ 64 }, i8, s_ref_in_3_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_3_f_zp, s_ref_in_3_f_scale));
  Tensor out = new RamTensor({ 64 }, flt);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_3_q_zp, s_out_3_q_scale));
  Tensor out_ref = new RomTensor({ 64 }, flt, s_ref_out_3_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_3_f_zp, s_ref_out_3_f_scale));

  pTensor::ReferenceOperators::LogisticOperator<int8_t> logisticOp;
  logisticOp
  .set_inputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::in, in }
  }).set_outputs({ 
    { pTensor::ReferenceOperators::LogisticOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 64; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 2);
}
}

