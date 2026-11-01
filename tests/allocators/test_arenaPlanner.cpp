// Arena memory planning: lifetimes, placement, in-place sharing, and running
// operators with their output written over their input.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

// Checks the invariant the planner exists for: any two tensors that are live
// at the same time, and do not legitimately share a buffer, have disjoint
// bytes. Also checks alignment and that everything fits in arena_bytes().
void expect_valid(const ArenaPlannerBase& p) {
  const uint16_t n = p.num_tensors();
  // Which buffer each tensor ended up in.
  std::vector<uint16_t> group(n);
  for (uint16_t i = 0; i < n; ++i) {
    uint16_t r = i;
    while (p.shares_buffer(r)) r = p.entry(r).alias;
    group[i] = r;
  }
  for (uint16_t i = 0; i < n; ++i) {
    const ArenaPlannerBase::Entry& a = p.entry(i);
    EXPECT_EQ(a.offset % p.alignment(), 0u) << "tensor " << i;
    EXPECT_LE(a.offset + a.size, p.arena_bytes()) << "tensor " << i;
    for (uint16_t j = i + 1; j < n; ++j) {
      const ArenaPlannerBase::Entry& b = p.entry(j);
      if (a.size == 0 || b.size == 0) continue;
      const bool live_together = a.first <= b.last && b.first <= a.last;
      if (!live_together) continue;
      if (group[i] == group[j]) {
        // Sharing is only legal between an in-place op's input and output:
        // the input's last use is the op that first writes the output.
        const bool handover = (b.alias == i && a.last == b.first) ||
                              (a.alias == j && b.last == a.first);
        EXPECT_TRUE(handover) << "tensors " << i << " and " << j
                              << " share a buffer while both live";
        continue;
      }
      const bool disjoint =
          a.offset + a.size <= b.offset || b.offset + b.size <= a.offset;
      EXPECT_TRUE(disjoint) << "tensors " << i << " and " << j << " overlap";
    }
  }
  EXPECT_LE(p.arena_bytes(), p.naive_bytes());
}

}  // namespace

TEST(ArenaPlanner, chain_reuses_dead_buffers) {
  // in -> A -> B -> C -> out, each op reads one tensor and writes the next.
  ArenaPlanner<8, 4> p;
  const uint16_t t0 = p.add_tensor(100);
  const uint16_t t1 = p.add_tensor(400);
  const uint16_t t2 = p.add_tensor(200);
  const uint16_t t3 = p.add_tensor(300);
  const uint16_t t4 = p.add_tensor(40);
  p.add_op(t0, t1);
  p.add_op(t1, t2);
  p.add_op(t2, t3);
  p.add_op(t3, t4);
  p.mark_output(t4);
  ASSERT_TRUE(p.plan());
  expect_valid(p);
  EXPECT_EQ(p.naive_bytes(), 1040u);
  // The worst op is t1 -> t2 or t2 -> t3: 600 or 500 bytes live.
  EXPECT_EQ(p.live_peak_bytes(), 600u);
  EXPECT_EQ(p.arena_bytes(), 600u);
  EXPECT_EQ(p.inplace_count(), 0);
}

TEST(ArenaPlanner, lifetimes) {
  ArenaPlanner<4> p;
  const uint16_t in = p.add_tensor(16);
  const uint16_t mid = p.add_tensor(16);
  const uint16_t out = p.add_tensor(16);
  p.add_op(in, mid);
  p.add_op(mid, out);
  p.mark_output(out);
  p.plan();
  EXPECT_EQ(p.entry(in).first, -1);   // a graph input exists before op 0
  EXPECT_EQ(p.entry(in).last, 0);
  EXPECT_EQ(p.entry(mid).first, 0);
  EXPECT_EQ(p.entry(mid).last, 1);
  EXPECT_EQ(p.entry(out).first, 1);
  EXPECT_EQ(p.entry(out).last, 2);    // a graph output outlives the last op
  // in and out are never live together, so they share offset 0.
  EXPECT_EQ(p.offset(in), p.offset(out));
}

TEST(ArenaPlanner, skip_connection_keeps_tensor_alive) {
  // x -> a -> b; y = add(b, x). x must survive until the add.
  ArenaPlanner<8, 4> p;
  const uint16_t x = p.add_tensor(64);
  const uint16_t a = p.add_tensor(64);
  const uint16_t b = p.add_tensor(64);
  const uint16_t y = p.add_tensor(64);
  p.add_op(x, a);
  p.add_op(a, b);
  const uint16_t in[2] = {b, x};
  p.add_inplace_op(in, 2, y);
  p.mark_output(y);
  p.plan();
  expect_valid(p);
  EXPECT_EQ(p.entry(x).last, 2);
  EXPECT_NE(p.offset(x), p.offset(b));
  // The add is b's last reader, so y may take over b's bytes.
  EXPECT_TRUE(p.shares_buffer(y));
  EXPECT_EQ(p.offset(y), p.offset(b));
  EXPECT_EQ(p.arena_bytes(), 3 * 64u);
}

TEST(ArenaPlanner, inplace_chain_folds_into_one_buffer) {
  // conv -> relu -> clip -> scale, all elementwise after the conv.
  ArenaPlanner<8, 4> p;
  const uint16_t x = p.add_tensor(100);
  const uint16_t c = p.add_tensor(400);
  const uint16_t r = p.add_tensor(400);
  const uint16_t k = p.add_tensor(400);
  const uint16_t s = p.add_tensor(400);
  p.add_op(x, c);
  p.add_inplace_op(c, r);
  p.add_inplace_op(r, k);
  p.add_inplace_op(k, s);
  p.mark_output(s);
  p.plan();
  expect_valid(p);
  EXPECT_EQ(p.inplace_count(), 3);
  EXPECT_EQ(p.offset(r), p.offset(c));
  EXPECT_EQ(p.offset(k), p.offset(c));
  EXPECT_EQ(p.offset(s), p.offset(c));
  EXPECT_EQ(p.live_peak_bytes(), 800u);  // without sharing
  EXPECT_EQ(p.arena_bytes(), 500u);      // with sharing
}

TEST(ArenaPlanner, inplace_refused_when_input_read_later) {
  ArenaPlanner<8, 4> p;
  const uint16_t x = p.add_tensor(32);
  const uint16_t r = p.add_tensor(32);
  const uint16_t z = p.add_tensor(32);
  p.add_inplace_op(x, r);  // relu(x), but x is read again below
  const uint16_t in[2] = {r, x};
  p.add_op(in, 2, &z, 1);
  p.mark_output(z);
  p.plan();
  expect_valid(p);
  EXPECT_FALSE(p.shares_buffer(r));
  EXPECT_NE(p.offset(r), p.offset(x));
}

TEST(ArenaPlanner, inplace_refused_for_graph_output_or_larger_output) {
  ArenaPlanner<8, 4> p;
  const uint16_t x = p.add_tensor(32);
  const uint16_t y = p.add_tensor(32);
  p.add_inplace_op(x, y);
  p.mark_output(x);  // caller still wants x afterwards
  p.mark_output(y);
  const uint16_t u = p.add_tensor(8);
  const uint16_t v = p.add_tensor(16);
  p.add_op(y, u);
  p.add_inplace_op(u, v);  // output bigger than the input it would overwrite
  p.mark_output(v);
  p.plan();
  expect_valid(p);
  EXPECT_FALSE(p.shares_buffer(y));
  EXPECT_FALSE(p.shares_buffer(v));
  EXPECT_EQ(p.inplace_count(), 0);
}

TEST(ArenaPlanner, narrowing_inplace_is_allowed) {
  // float -> int8 quantize writes byte i after reading bytes 4i..4i+3.
  ArenaPlanner<4, 4> p;
  const uint16_t f = p.add_tensor(40);
  const uint16_t q = p.add_tensor(10);
  p.add_inplace_op(f, q);
  p.mark_output(q);
  p.plan();
  EXPECT_TRUE(p.shares_buffer(q));
  EXPECT_EQ(p.arena_bytes(), 40u);
}

TEST(ArenaPlanner, alignment_and_table_full) {
  ArenaPlanner<2, 16> p;
  const uint16_t a = p.add_tensor(3);
  const uint16_t b = p.add_tensor(5);
  EXPECT_EQ(p.add_tensor(1), ArenaPlannerBase::kInvalid);
  EXPECT_EQ(p.add_op(a, 7), -1);
  p.add_op(a, b);
  p.mark_output(b);
  p.plan();
  expect_valid(p);
  EXPECT_EQ(p.arena_bytes(), 32u);
  EXPECT_EQ(p.offset(a) % 16, 0u);
  EXPECT_EQ(p.offset(b) % 16, 0u);
}

TEST(ArenaPlanner, unused_tensor_reported) {
  ArenaPlanner<4> p;
  const uint16_t a = p.add_tensor(8);
  const uint16_t b = p.add_tensor(8);
  p.add_tensor(8);  // never used
  p.add_op(a, b);
  EXPECT_FALSE(p.plan());
}

TEST(ArenaPlanner, replanning_is_stable) {
  ArenaPlanner<4> p;
  const uint16_t a = p.add_tensor(64);
  const uint16_t b = p.add_tensor(64);
  p.add_inplace_op(a, b);
  p.mark_output(b);
  p.plan();
  const uint32_t first = p.arena_bytes();
  p.plan();
  EXPECT_EQ(p.arena_bytes(), first);
  EXPECT_EQ(p.inplace_count(), 1);
  p.reset();
  EXPECT_EQ(p.num_tensors(), 0);
}

TEST(ArenaPlanner, random_graphs_never_overlap) {
  srand(1234);
  for (int trial = 0; trial < 300; ++trial) {
    ArenaPlanner<40, 8> p;
    const int n_inputs = 1 + rand() % 3;
    std::vector<uint16_t> made;
    for (int i = 0; i < n_inputs; ++i) made.push_back(p.add_tensor(1 + rand() % 500));
    const int n_ops = 1 + rand() % 30;
    for (int o = 0; o < n_ops; ++o) {
      const int k = 1 + rand() % 2;
      uint16_t in[2];
      for (int i = 0; i < k; ++i) in[i] = made[rand() % made.size()];
      const uint16_t out = p.add_tensor(1 + rand() % 500);
      if (rand() % 2)
        p.add_inplace_op(in, k, out);
      else
        p.add_op(in, k, &out, 1);
      made.push_back(out);
      if (rand() % 8 == 0) p.mark_output(out);
    }
    p.mark_output(made.back());
    p.plan();
    expect_valid(p);
    if (::testing::Test::HasFailure()) {
      ADD_FAILURE() << "trial " << trial;
      return;
    }
    if (p.inplace_count() == 0) EXPECT_GE(p.arena_bytes(), p.live_peak_bytes());
  }
}

TEST(ArenaAllocatorHighWater, tracks_furthest_block) {
  localCircularArenaAllocator<1024, uint32_t> a;
  EXPECT_EQ(a.high_water_mark(), 0u);
  void* x = a.allocate(100);
  void* y = a.allocate(200);
  EXPECT_GE(a.high_water_mark(), 300u);
  const size_t hw = a.high_water_mark();
  a.deallocate(y);
  a.deallocate(x);
  void* z = a.allocate(50);  // reuses the first region
  (void)z;
  EXPECT_EQ(a.high_water_mark(), hw);
  a.reset_high_water_mark();
  EXPECT_EQ(a.high_water_mark(), 0u);
}

// ---- operators run with output aliased onto the input -----------------------

namespace {

struct InPlaceFixture : public ::testing::Test {
  localCircularArenaAllocator<4096> meta_allocator;
  localCircularArenaAllocator<4096, uint32_t> ram_allocator;
  SimpleErrorHandler err{10};
  int errors = 0;
  void SetUp() override {
    err.set_onError([this](Error*) { ++errors; });
    Context::get_default_context()->set_ErrorHandler(&err);
    Context::get_default_context()->set_metadata_allocator(&meta_allocator);
    Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  }
  void TearDown() override { err.set_onError([](Error*) {}); }
};

const float kInput[12] = {-3.0f, -1.5f, -0.25f, 0.0f,  0.5f,  1.0f,
                          2.0f,  4.0f,  7.5f,   -6.0f, 0.75f, 3.25f};

// Runs a one-input one-output operator twice, once into a separate buffer and
// once with the output written over the input, and compares.
template <typename Op>
void check_unary_inplace(Op& separate_op, Op& inplace_op) {
  float a[12], b[12], out[12];
  memcpy(a, kInput, sizeof a);
  memcpy(b, kInput, sizeof b);
  {
    Tensor x = new BufferTensor({3, 4}, flt, a);
    Tensor y = new BufferTensor({3, 4}, flt, out);
    separate_op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
  }
  {
    Tensor x = new BufferTensor({3, 4}, flt, b);
    Tensor y = new BufferTensor({3, 4}, flt, b);
    inplace_op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
  }
  for (int i = 0; i < 12; ++i) EXPECT_EQ(b[i], out[i]) << i;
}

}  // namespace

TEST_F(InPlaceFixture, relu_float) {
  ReLUOperator<float> o1, o2;
  check_unary_inplace(o1, o2);
}

TEST_F(InPlaceFixture, softmax_float) {
  // Row-wise: every element of a row is read before the row is rewritten.
  SoftmaxOperator<float> o1, o2;
  check_unary_inplace(o1, o2);
}

TEST_F(InPlaceFixture, relu6_float) {
  ReLU6Operator<float> o1, o2;
  check_unary_inplace(o1, o2);
}

TEST_F(InPlaceFixture, elementwise_activations) {
  { LeakyReLUOperator o1, o2; check_unary_inplace(o1, o2); }
  { HardSwishOperator o1, o2; check_unary_inplace(o1, o2); }
  { SwishOperator o1, o2; check_unary_inplace(o1, o2); }
  { GeluOperator o1, o2; check_unary_inplace(o1, o2); }
  { ClipOperator o1, o2; check_unary_inplace(o1, o2); }
  EXPECT_EQ(errors, 0);
}

TEST_F(InPlaceFixture, broadcast_add_over_first_input) {
  const float bias[4] = {1.0f, -2.0f, 0.5f, 10.0f};
  float a[12], out[12];
  memcpy(a, kInput, sizeof a);
  typedef BroadcastAddOperator<float> Add;
  {
    Tensor x = new BufferTensor({3, 4}, flt, a);
    Tensor bb = new RomTensor({4}, flt, bias);
    Tensor y = new BufferTensor({3, 4}, flt, out);
    Add op;
    op.set_inputs({{Add::a, x}, {Add::b, bb}}).set_outputs({{Add::c, y}}).eval();
  }
  {
    Tensor x = new BufferTensor({3, 4}, flt, a);
    Tensor bb = new RomTensor({4}, flt, bias);
    Tensor y = new BufferTensor({3, 4}, flt, a);
    Add op;
    op.set_inputs({{Add::a, x}, {Add::b, bb}}).set_outputs({{Add::c, y}}).eval();
  }
  for (int i = 0; i < 12; ++i) EXPECT_EQ(a[i], out[i]) << i;
}

TEST_F(InPlaceFixture, int8_relu_and_quantize_narrowing) {
  // int8 ReLU in place.
  int8_t q[8] = {-100, -1, 0, 1, 50, 127, -128, 3};
  int8_t expect[8];
  for (int i = 0; i < 8; ++i) expect[i] = q[i] < 0 ? 0 : q[i];
  {
    Tensor x = new BufferTensor({8}, i8, q);
    x->set_quantization_params(PerTensorQuantizationParams(0, 0.5f));
    Tensor y = new BufferTensor({8}, i8, q);
    y->set_quantization_params(PerTensorQuantizationParams(0, 0.5f));
    ReLUOperator<int8_t> op;
    op.set_inputs({{ReLUOperator<int8_t>::in, x}})
        .set_outputs({{ReLUOperator<int8_t>::out, y}}).eval();
  }
  for (int i = 0; i < 8; ++i) EXPECT_EQ(q[i], expect[i]) << i;

#if PTENSOR_WITH_QUANTIZED_OPS
  // float -> int8 quantize written over the float input's first bytes.
  float buf[12];
  memcpy(buf, kInput, sizeof buf);
  int8_t ref[12];
  typedef TflmSymQuantOps::QuantizeOperator<int8_t, float> Q;
  {
    Tensor x = new RomTensor({12}, flt, kInput);
    Tensor y = new BufferTensor({12}, i8, ref);
    y->set_quantization_params(PerTensorQuantizationParams(3, 0.05f));
    Q op;
    op.set_inputs({{Q::input, x}}).set_outputs({{Q::output, y}}).eval();
  }
  {
    Tensor x = new BufferTensor({12}, flt, buf);
    Tensor y = new BufferTensor({12}, i8, buf);
    y->set_quantization_params(PerTensorQuantizationParams(3, 0.05f));
    Q op;
    op.set_inputs({{Q::input, x}}).set_outputs({{Q::output, y}}).eval();
  }
  EXPECT_EQ(memcmp(buf, ref, sizeof ref), 0);
#endif
  EXPECT_EQ(errors, 0);
}

// A small conv network run twice: every activation in its own buffer, and in
// one planned arena with in-place ReLUs. Results must be bit-identical.
TEST_F(InPlaceFixture, planned_network_matches_unplanned) {
  // input 6x6x1 -> conv 3x3x4 (SAME) -> relu -> maxpool 2 -> conv 3x3x2 -> relu
  float input[36], w1[4 * 3 * 3 * 1], b1[4], w2[2 * 3 * 3 * 4], b2[2];
  srand(7);
  for (float& v : input) v = (rand() % 200 - 100) / 50.0f;
  for (float& v : w1) v = (rand() % 200 - 100) / 100.0f;
  for (float& v : b1) v = (rand() % 200 - 100) / 100.0f;
  for (float& v : w2) v = (rand() % 200 - 100) / 100.0f;
  for (float& v : b2) v = (rand() % 200 - 100) / 100.0f;

  const uint32_t n_c1 = 6 * 6 * 4, n_p = 3 * 3 * 4, n_c2 = 3 * 3 * 2;
  typedef Conv2dOperator<float> Conv;
  typedef ReLUOperator<float> Relu;
  typedef MaxPoolOperator<float> Pool;

  auto run = [&](float* x, float* c1, float* r1, float* p1, float* c2,
                 float* r2) {
    memcpy(x, input, sizeof input);
    Tensor tx = new BufferTensor({1, 6, 6, 1}, flt, x);
    Tensor tw1 = new RomTensor({4, 3, 3, 1}, flt, w1);
    Tensor tb1 = new RomTensor({4}, flt, b1);
    Tensor tc1 = new BufferTensor({1, 6, 6, 4}, flt, c1);
    Conv conv1({1, 1}, SAME);
    conv1.set_inputs({{Conv::in, tx}, {Conv::filter, tw1}, {Conv::bias, tb1}})
        .set_outputs({{Conv::out, tc1}}).eval();
    Tensor tr1 = new BufferTensor({1, 6, 6, 4}, flt, r1);
    Relu relu1;
    relu1.set_inputs({{Relu::in, tc1}}).set_outputs({{Relu::out, tr1}}).eval();
    Tensor tp1 = new BufferTensor({1, 3, 3, 4}, flt, p1);
    Pool pool({2, 2}, {1, 2, 2, 1}, VALID);
    pool.set_inputs({{Pool::in, tr1}}).set_outputs({{Pool::out, tp1}}).eval();
    Tensor tw2 = new RomTensor({2, 3, 3, 4}, flt, w2);
    Tensor tb2 = new RomTensor({2}, flt, b2);
    Tensor tc2 = new BufferTensor({1, 3, 3, 2}, flt, c2);
    Conv conv2({1, 1}, SAME);
    conv2.set_inputs({{Conv::in, tp1}, {Conv::filter, tw2}, {Conv::bias, tb2}})
        .set_outputs({{Conv::out, tc2}}).eval();
    Tensor tr2 = new BufferTensor({1, 3, 3, 2}, flt, r2);
    Relu relu2;
    relu2.set_inputs({{Relu::in, tc2}}).set_outputs({{Relu::out, tr2}}).eval();
  };

  // Unplanned: separate buffers.
  float x[36], c1[n_c1], r1[n_c1], p1[n_p], c2[n_c2], r2[n_c2];
  run(x, c1, r1, p1, c2, r2);

  ArenaPlanner<8> plan;
  const uint16_t ix = plan.add_tensor(sizeof x);
  const uint16_t ic1 = plan.add_tensor(sizeof c1);
  const uint16_t ir1 = plan.add_tensor(sizeof r1);
  const uint16_t ip1 = plan.add_tensor(sizeof p1);
  const uint16_t ic2 = plan.add_tensor(sizeof c2);
  const uint16_t ir2 = plan.add_tensor(sizeof r2);
  plan.add_op(ix, ic1);
  plan.add_inplace_op(ic1, ir1);
  plan.add_op(ir1, ip1);
  plan.add_op(ip1, ic2);
  plan.add_inplace_op(ic2, ir2);
  plan.mark_output(ir2);
  ASSERT_TRUE(plan.plan());
  expect_valid(plan);
  EXPECT_EQ(plan.inplace_count(), 2);
  EXPECT_LT(plan.arena_bytes(), plan.naive_bytes());

  std::vector<uint8_t> arena(plan.arena_bytes(), 0xCD);
  float* px = static_cast<float*>(plan.place(ix, arena.data()));
  float* pc1 = static_cast<float*>(plan.place(ic1, arena.data()));
  float* pr1 = static_cast<float*>(plan.place(ir1, arena.data()));
  float* pp1 = static_cast<float*>(plan.place(ip1, arena.data()));
  float* pc2 = static_cast<float*>(plan.place(ic2, arena.data()));
  float* pr2 = static_cast<float*>(plan.place(ir2, arena.data()));
  run(px, pc1, pr1, pp1, pc2, pr2);
  EXPECT_EQ(memcmp(pr2, r2, sizeof r2), 0);
  EXPECT_EQ(errors, 0);
}

// Reassigning a Tensor handle must free the tensor it held, or a model that
// keeps one handle per layer fills the RAM arena after a few inferences.
TEST_F(InPlaceFixture, tensor_reassignment_frees_previous_tensor) {
  {
    Tensor cur = new RamTensor({256}, i8);
    for (int i = 0; i < 200; ++i) {
      Tensor next = new RamTensor({256}, i8);
      cur = std::move(next);
      cur = new RamTensor({256}, i8);
    }
    EXPECT_LE(ram_allocator.num_blocks(), 4u);
    EXPECT_LE(meta_allocator.num_blocks(), 8u);
  }
  EXPECT_EQ(errors, 0);
}
