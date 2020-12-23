// Auto generated by utensor-cli

#include "src/pTensor/ops/ArrayOps.hpp"
#include "qmul0_weight.hpp"
#include "src/pTensor/core/tensor.hpp"
#include "src/pTensor/ops/MathOps.hpp"
#include "qmul0.hpp"
#include "src/pTensor/core/context.hpp"


void get_qmul0_ctx(Context& ctx) {
{    
    ctx.add(new BinaryTensor<float>({10}, inline_ref_0_0), 
            "ref_0:0");
}
{    
    ctx.add(new BinaryTensor<float>({10}, inline_a_0_0), 
            "a_0:0", 
            2);
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_c_0_eightbit_a_0__port__0_reshape_dims_0), 
            "c_0_eightbit/a_0__port__0/reshape_dims:0", 
            1);
}
{
    ctx.add(new RamTensor<float>(), "c_0_eightbit/b_0__port__0/reshape:0", 2);
    ctx.push(new ReshapeOp(), 
             { "a_0:0", "c_0_eightbit/a_0__port__0/reshape_dims:0" },
             { "c_0_eightbit/b_0__port__0/reshape:0" });
    ctx.eval();
}
{    
    ctx.add(new BinaryTensor<int>({1}, inline_c_0_eightbit_a_0__port__0_reduction_dims_0), 
            "c_0_eightbit/a_0__port__0/reduction_dims:0", 
            2);
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "c_0_eightbit/b_0__port__0/min:0", 1);
    ctx.push(new MinOp(), 
             { "c_0_eightbit/b_0__port__0/reshape:0", "c_0_eightbit/a_0__port__0/reduction_dims:0" },
             { "c_0_eightbit/b_0__port__0/min:0" });
    ctx.eval();
}
{   
    RamTensor<float>* out_tensor;
    out_tensor = new RamTensor<float>({ 1 });
    ctx.add(out_tensor, "c_0_eightbit/b_0__port__0/max:0", 1);
    ctx.push(new MaxOp(), 
             { "c_0_eightbit/b_0__port__0/reshape:0", "c_0_eightbit/a_0__port__0/reduction_dims:0" },
             { "c_0_eightbit/b_0__port__0/max:0" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<uint8_t>(), "c_0_eightbit/b_0__port__0/quantize:0", 2);
    ctx.add(new RamTensor<float>({1}), "c_0_eightbit/b_0__port__0/quantize:1", 2);
    ctx.add(new RamTensor<float>({1}), "c_0_eightbit/b_0__port__0/quantize:2", 2);
    ctx.push(new QuantizeV2Op(),
             {  "a_0:0",  "c_0_eightbit/b_0__port__0/min:0", "c_0_eightbit/b_0__port__0/max:0" },
             {  "c_0_eightbit/b_0__port__0/quantize:0",  "c_0_eightbit/b_0__port__0/quantize:1", "c_0_eightbit/b_0__port__0/quantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<int>(), "c_0/eightbit:0", 2);
    ctx.add(new RamTensor<float>({1}), "c_0/eightbit:1", 2);
    ctx.add(new RamTensor<float>({1}), "c_0/eightbit:2", 2);
    ctx.push(new QuantizedMulOp<uint8_t, uint8_t, int>(), 
             { "c_0_eightbit/b_0__port__0/quantize:0", "c_0_eightbit/b_0__port__0/quantize:1", "c_0_eightbit/b_0__port__0/quantize:2", "c_0_eightbit/b_0__port__0/quantize:0", "c_0_eightbit/b_0__port__0/quantize:1",  "c_0_eightbit/b_0__port__0/quantize:2" },
             { "c_0/eightbit:0", "c_0/eightbit:1",  "c_0/eightbit:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>({1}), "c_0/eightbit/requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "c_0/eightbit/requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "c_0/eightbit:0", "c_0/eightbit:1", "c_0/eightbit:2" },
             { "c_0/eightbit/requant_range:0", "c_0/eightbit/requant_range:1" });
    ctx.eval();
}
{   
    ctx.add(new RamTensor<uint8_t>(), "c_0/eightbit/requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "c_0/eightbit/requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "c_0/eightbit/requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "c_0/eightbit:0", "c_0/eightbit:1", "c_0/eightbit:2", "c_0/eightbit/requant_range:0", "c_0/eightbit/requant_range:1" },
             { "c_0/eightbit/requantize:0", "c_0/eightbit/requantize:1", "c_0/eightbit/requantize:2" });
    ctx.eval();
}
{
    ctx.add(new RamTensor<float>(), "c_0:0");
    ctx.push(new DequantizeOp(), 
             { "c_0/eightbit/requantize:0", "c_0/eightbit/requantize:1", "c_0/eightbit/requantize:2" },
             { "c_0:0" });
}
}
