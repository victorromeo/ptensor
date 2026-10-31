// Second translation unit that includes the umbrella header. Linking this
// together with test_top_include.cpp fails if any header defines a
// non-inline function (an ODR violation), which would break every user
// project that includes pTensor.h from more than one source file.
#include "pTensor.h"

int ptensor_second_tu_marker() { return 42; }
