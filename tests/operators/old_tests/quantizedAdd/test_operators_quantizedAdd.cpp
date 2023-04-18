#include "test_helper.h"
#include "models/add_graph.hpp"
#include "src/pTensor/loaders/tensorIdxImporter.hpp"

#include <iostream>
using std::cout;
using std::endl;

TensorIdxImporter t_import;


// Default to using GTest like asserts and expects as these give more info that unity
// We will forward these commands to unity in test_helper.h
void test_operators_quantizedAdd(){
    Context ctx;
    get_add_graph_ctx(ctx);
    S_TENSOR output_z = ctx.get("z:0");
    ctx.eval();

    Tensor* ref_z = t_import.float_import("/fs/constants/add_graph/ref_z.idx");

    // compare the results
    double err = meanAbsErr<float>(ref_z, output_z);

    cout << "err: " << err << endl;
    EXPECT_LT(err , 0.0003);
}


// First configure the pTensor test runner
PTENSOR_TEST_CONFIGURE()

// Second declare tests to run
PTENSOR_TEST(operators, quantizedAdd, "Generated test for quantized Add")


// Third, run like hell
PTENSOR_TEST_RUN()
