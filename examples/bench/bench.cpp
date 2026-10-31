// Host micro-benchmark for the float kernels. Build with -DPTENSOR_BENCH=ON
// and run build/examples/bench/ptensor_bench. Each line prints the best time
// of several runs, so numbers are comparable between builds.
#include <chrono>
#include <cstdio>
#include <cstring>

#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

localCircularArenaAllocator<8192, uint32_t, 64> meta;
localCircularArenaAllocator<1024, uint32_t, 16> ram;
SimpleErrorHandler err(4);

float buf_a[1 << 16], buf_b[1 << 16], buf_c[1 << 16], buf_d[1 << 16];
float sink = 0;

void fill(float* p, size_t n, float s) {
  uint32_t x = 12345u + static_cast<uint32_t>(s * 1000);
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    p[i] = (static_cast<float>(x >> 8) / 16777216.0f - 0.5f) * s;
  }
}

template <typename F>
void bench(const char* name, double flops, F fn) {
  double best = 1e30;
  int reps = 0;
  auto start = std::chrono::steady_clock::now();
  do {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (us < best) best = us;
    ++reps;
  } while (reps < 3 ||
           (reps < 200 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.5));
  sink += buf_c[0];
  double cs = 0;
  for (int i = 0; i < 4096; ++i) cs += buf_c[i] * (1 + (i & 7));
  std::printf("%-34s %10.1f us  %7.2f GFLOP/s  [check %.6g]\n", name, best, flops / best / 1e3, cs);
}

}  // namespace

int main() {
  Context::get_default_context()->set_metadata_allocator(&meta);
  Context::get_default_context()->set_ram_data_allocator(&ram);
  Context::get_default_context()->set_ErrorHandler(&err);
  fill(buf_a, 1 << 16, 1.0f);
  fill(buf_b, 1 << 16, 0.5f);
  fill(buf_d, 1 << 16, 0.25f);

  bench("matmul 64x128 @ 128x64", 2.0 * 64 * 128 * 64, [] {
    Tensor a = new RomTensor({64, 128}, flt, buf_a);
    Tensor b = new RomTensor({128, 64}, flt, buf_b);
    Tensor c = new BufferTensor({64, 64}, flt, buf_c);
    MatrixMultOperator<float> op;
    op.set_inputs({{MatrixMultOperator<float>::a, a}, {MatrixMultOperator<float>::b, b}})
        .set_outputs({{MatrixMultOperator<float>::c, c}}).eval();
  });

  bench("fully_connected 1x512 -> 256 relu", 2.0 * 512 * 256, [] {
    typedef FullyConnectedOperator<float> FC;
    Tensor x = new RomTensor({1, 512}, flt, buf_a);
    Tensor w = new RomTensor({512, 256}, flt, buf_b);
    Tensor b = new RomTensor({256}, flt, buf_d);
    Tensor y = new BufferTensor({1, 256}, flt, buf_c);
    FC op(Fuseable::ReLU<float>);
    op.set_inputs({{FC::input, x}, {FC::filter, w}, {FC::bias, b}})
        .set_outputs({{FC::output, y}}).eval();
  });

  bench("conv2d 1x32x32x16 3x3 -> 32 SAME", 2.0 * 32 * 32 * 32 * 9 * 16, [] {
    typedef Conv2dOperator<float> Op;
    Tensor x = new RomTensor({1, 32, 32, 16}, flt, buf_a);
    Tensor w = new RomTensor({32, 3, 3, 16}, flt, buf_b);
    Tensor b = new RomTensor({32}, flt, buf_d);
    Tensor y = new BufferTensor({1, 32, 32, 32}, flt, buf_c);
    Op op({1, 1}, SAME);
    op.set_inputs({{Op::in, x}, {Op::filter, w}, {Op::bias, b}})
        .set_outputs({{Op::out, y}}).eval();
  });

  bench("depthwise 1x32x32x32 3x3 SAME", 2.0 * 32 * 32 * 32 * 9, [] {
    typedef DepthwiseConv2dOperator<float> Op;
    Tensor x = new RomTensor({1, 32, 32, 32}, flt, buf_a);
    Tensor w = new RomTensor({1, 3, 3, 32}, flt, buf_b);
    Tensor b = new RomTensor({32}, flt, buf_d);
    Tensor y = new BufferTensor({1, 32, 32, 32}, flt, buf_c);
    Op op(1, 1, SAME, 1, 1, 1);
    op.set_inputs({{Op::in, x}, {Op::filter, w}, {Op::bias, b}})
        .set_outputs({{Op::out, y}}).eval();
  });

  bench("maxpool 1x32x32x32 2x2/2", 32.0 * 16 * 16 * 4, [] {
    typedef MaxPoolOperator<float> Op;
    Tensor x = new RomTensor({1, 32, 32, 32}, flt, buf_a);
    Tensor y = new BufferTensor({1, 16, 16, 32}, flt, buf_c);
    Op op({2, 2}, {1, 2, 2, 1}, VALID);
    op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
  });

  bench("mul 1x32x32x64 * [64] broadcast", 65536.0, [] {
    typedef MulOperator<float> Op;
    Tensor a = new RomTensor({1, 32, 32, 64}, flt, buf_a);
    Tensor b = new RomTensor({64}, flt, buf_b);
    Tensor c = new BufferTensor({1, 32, 32, 64}, flt, buf_c);
    Op op;
    op.set_inputs({{Op::a, a}, {Op::b, b}}).set_outputs({{Op::c, c}}).eval();
  });

  bench("add 65536 (equal shapes)", 65536.0, [] {
    typedef AddOperator<float> Op;
    Tensor a = new RomTensor({65535}, flt, buf_a);
    Tensor b = new RomTensor({65535}, flt, buf_b);
    Tensor c = new BufferTensor({65535}, flt, buf_c);
    Op op;
    op.set_inputs({{Op::a, a}, {Op::b, b}}).set_outputs({{Op::c, c}}).eval();
  });

  bench("swish 65535", 65535.0, [] {
    typedef SwishOperator Op;
    Tensor a = new RomTensor({65535}, flt, buf_a);
    Tensor c = new BufferTensor({65535}, flt, buf_c);
    Op op;
    op.set_inputs({{Op::in, a}}).set_outputs({{Op::out, c}}).eval();
  });

  bench("lstm cell in=64 hidden=128", 2.0 * (64 + 128) * 512, [] {
    typedef LstmCellOperator Op;
    Tensor x = new RomTensor({1, 64}, flt, buf_a);
    Tensor h = new RomTensor({1, 128}, flt, buf_a + 64);
    Tensor c = new RomTensor({1, 128}, flt, buf_a + 192);
    Tensor W = new RomTensor({64, 512}, flt, buf_b);
    Tensor U = new RomTensor({128, 512}, flt, buf_b + 32768);
    Tensor b = new RomTensor({512}, flt, buf_d);
    Tensor ho = new BufferTensor({1, 128}, flt, buf_c);
    Tensor co = new BufferTensor({1, 128}, flt, buf_c + 128);
    Op op;
    op.set_inputs({{Op::x, x}, {Op::h, h}, {Op::c, c}, {Op::W, W}, {Op::U, U}, {Op::b, b}})
        .set_outputs({{Op::h_out, ho}, {Op::c_out, co}}).eval();
  });

  std::printf("(checksum %g)\n", sink);
  return 0;
}
