
#include <cstring>
#include <iostream>

#include "ReLuOperator.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "constants_relu.hpp"
#include "constants_int8_relu.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
using std::cout;
using std::endl;

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_float_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, flt);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_float_0[i];
  }

  InPlaceReLU<float> ReLU;
  ReLU.set_inputs({{InPlaceReLU<float>::x, io}}).eval();

  float tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_float_0[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_float_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, flt);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_float_1[i];
  }

  InPlaceReLU<float> ReLU;
  ReLU.set_inputs({{InPlaceReLU<float>::x, io}}).eval();

  float tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_float_1[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_float_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, flt);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_float_2[i];
  }

  InPlaceReLU<float> ReLU;
  ReLU.set_inputs({{InPlaceReLU<float>::x, io}}).eval();

  float tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_float_2[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_float_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, flt);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_float_3[i];
  }

  InPlaceReLU<float> ReLU;
  ReLU.set_inputs({{InPlaceReLU<float>::x, io}}).eval();

  float tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_float_3[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_float_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, flt);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_float_4[i];
  }

  InPlaceReLU<float> ReLU;
  ReLU.set_inputs({{InPlaceReLU<float>::x, io}}).eval();

  float tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_float_4[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int8_t_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i8);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int8_t_0[i];
  }

  InPlaceReLU<int8_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int8_t>::x, io}}).eval();

  int8_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int8_t_0[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int8_t_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i8);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int8_t_1[i];
  }

  InPlaceReLU<int8_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int8_t>::x, io}}).eval();

  int8_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int8_t_1[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int8_t_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i8);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int8_t_2[i];
  }

  InPlaceReLU<int8_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int8_t>::x, io}}).eval();

  int8_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int8_t_2[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int8_t_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i8);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int8_t_3[i];
  }

  InPlaceReLU<int8_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int8_t>::x, io}}).eval();

  int8_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int8_t_3[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int8_t_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i8);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int8_t_4[i];
  }

  InPlaceReLU<int8_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int8_t>::x, io}}).eval();

  int8_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int8_t_4[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int16_t_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int16_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i16);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int16_t_0[i];
  }

  InPlaceReLU<int16_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int16_t>::x, io}}).eval();

  int16_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int16_t_0[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int16_t_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int16_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i16);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int16_t_1[i];
  }

  InPlaceReLU<int16_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int16_t>::x, io}}).eval();

  int16_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int16_t_1[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int16_t_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int16_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i16);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int16_t_2[i];
  }

  InPlaceReLU<int16_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int16_t>::x, io}}).eval();

  int16_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int16_t_2[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int16_t_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int16_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i16);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int16_t_3[i];
  }

  InPlaceReLU<int16_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int16_t>::x, io}}).eval();

  int16_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int16_t_3[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int16_t_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int16_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i16);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int16_t_4[i];
  }

  InPlaceReLU<int16_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int16_t>::x, io}}).eval();

  int16_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int16_t_4[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int32_t_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int32_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i32);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int32_t_0[i];
  }

  InPlaceReLU<int32_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int32_t>::x, io}}).eval();

  int32_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int32_t_0[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int32_t_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int32_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i32);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int32_t_1[i];
  }

  InPlaceReLU<int32_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int32_t>::x, io}}).eval();

  int32_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int32_t_1[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int32_t_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int32_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i32);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int32_t_2[i];
  }

  InPlaceReLU<int32_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int32_t>::x, io}}).eval();

  int32_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int32_t_2[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int32_t_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int32_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i32);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int32_t_3[i];
  }

  InPlaceReLU<int32_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int32_t>::x, io}}).eval();

  int32_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int32_t_3[i]), 0, 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(ReLU, random_inputs_int32_t_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784 * 2 * sizeof(int32_t), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({1, 28, 28, 1}, i32);
  for (int i = 0; i < 784; i++) {
    io(i) = s_in_int32_t_4[i];
  }

  InPlaceReLU<int32_t> ReLU;
  ReLU.set_inputs({{InPlaceReLU<int32_t>::x, io}}).eval();

  int32_t tmp;
  for (int i = 0; i < 784; i++) {
    tmp = io(i);
    EXPECT_NEAR((float)(tmp - s_ref_out_int32_t_4[i]), 0, 0.0001);
  }
}

/*********************************************
 * Quantized ReLU / ReLU6 vs TFLite
 * (constants_int8_relu.hpp holds TFLite 2.21's own outputs for every int8
 * input value; see gen_constants_int8_relu.py)
 *********************************************/

namespace {
template <typename Op>
void run_int8_relu(const int8_t* in, int8_t* out, uint16_t n, float si,
                   int32_t zi, float so, int32_t zo) {
  Tensor x = new RomTensor({n}, i8, in);
  Tensor y = new BufferTensor({n}, i8, out);
  x->set_quantization_params(PerTensorQuantizationParams(zi, si));
  y->set_quantization_params(PerTensorQuantizationParams(zo, so));
  Op op;
  op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
}
}  // namespace

TEST(ReLU, int8_matches_tflite) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<256, uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  int8_t in[256], out[256];
  for (int i = 0; i < 256; ++i) in[i] = static_cast<int8_t>(i - 128);
  run_int8_relu<ReLUOperator<int8_t> >(in, out, 256, relu_i8_in_scale, relu_i8_in_zp,
                                       relu_i8_out_scale, relu_i8_out_zp);
  for (int i = 0; i < 256; ++i) EXPECT_EQ(out[i], relu_i8_out[i]) << "input " << i - 128;
  run_int8_relu<ReLU6Operator<int8_t> >(in, out, 256, relu6_i8_in_scale, relu6_i8_in_zp,
                                        relu6_i8_out_scale, relu6_i8_out_zp);
  for (int i = 0; i < 256; ++i) EXPECT_EQ(out[i], relu6_i8_out[i]) << "input " << i - 128;
}

TEST(ReLU, int8_same_scale_is_exact_clamp) {
  // Same quantization in and out: ReLU must be max(q, zero_point) exactly.
  // The old float requantisation truncated, so e.g. scale 0.157 put about
  // 3% of values one step low.
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<256, uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  const float scales[3] = {0.157f, 0.05f, 0.0078125f};
  const int32_t zps[3] = {3, -128, -20};
  int8_t in[256], out[256];
  for (int i = 0; i < 256; ++i) in[i] = static_cast<int8_t>(i - 128);
  for (int k = 0; k < 3; ++k) {
    run_int8_relu<ReLUOperator<int8_t> >(in, out, 256, scales[k], zps[k], scales[k], zps[k]);
    for (int i = 0; i < 256; ++i)
      EXPECT_EQ(out[i], in[i] < zps[k] ? zps[k] : in[i]) << "scale " << scales[k] << " q " << int(in[i]);
  }
}
