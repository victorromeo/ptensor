// Minimal application used to measure pTensor's flash/RAM footprint.
// It runs out = relu(a * b + c) on 2x2 uint8 tensors with reference ops only.
#include "pTensor/allocators/arenaAllocator.hpp"
#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/ops/reference/AddOperator.hpp"
#include "pTensor/ops/reference/MatrixMultiplyOperator.hpp"
#include "pTensor/tensors/BufferTensor.hpp"
#include "pTensor/tensors/RamTensor.hpp"
#include "pTensor/tensors/RomTensor.hpp"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

static const uint8_t s_b[4] = {5, 6, 7, 8};
static const uint8_t s_c[4] = {1, 1, 1, 1};

static localCircularArenaAllocator<256> meta_allocator;
static localCircularArenaAllocator<64> ram_allocator;
static ErrorHandler error_handler;

volatile uint8_t g_input[4] = {1, 2, 3, 4};
volatile uint8_t g_output[4];

int main() {
  Context* ctx = Context::get_default_context();
  ctx->set_metadata_allocator(&meta_allocator);
  ctx->set_ram_data_allocator(&ram_allocator);
  ctx->set_ErrorHandler(&error_handler);

  uint8_t in_buf[4];
  uint8_t out_buf[4];
  for (int i = 0; i < 4; ++i) in_buf[i] = g_input[i];
  {
    Tensor a = new BufferTensor({2, 2}, u8, in_buf);
    Tensor b = new RomTensor({2, 2}, u8, s_b);
    Tensor c = new RomTensor({2, 2}, u8, s_c);
    Tensor tmp = new RamTensor({2, 2}, u8);
    Tensor out = new BufferTensor({2, 2}, u8, out_buf);

    MatrixMultOperator<uint8_t> mult;
    mult.set_inputs({{MatrixMultOperator<uint8_t>::a, a},
                     {MatrixMultOperator<uint8_t>::b, b}})
        .set_outputs({{MatrixMultOperator<uint8_t>::c, tmp}})
        .eval();
    AddOperator<uint8_t> add;
    add.set_inputs({{AddOperator<uint8_t>::a, tmp}, {AddOperator<uint8_t>::b, c}})
        .set_outputs({{AddOperator<uint8_t>::c, out}})
        .eval();
  }
  for (int i = 0; i < 4; ++i) g_output[i] = out_buf[i];
  return 0;
}
