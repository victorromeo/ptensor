#ifndef PTENSOR_OPERATOR_BASE_H
#define PTENSOR_OPERATOR_BASE_H
#include "TensorMap.hpp"
#include "context.hpp"
namespace pTensor {

enum Padding : uint8_t { UNKNOWN = 0, VALID = 1, SAME = 2 };

/**
 * Padding added before the first row / column of a windowed operator
 * (conv, depthwise conv, pooling), with TensorFlow semantics: VALID never
 * pads; SAME pads max(0, needed) / 2 in front (the odd extra row or column
 * goes after). `out` is the output size, `k` the kernel size.
 */
inline int32_t padding_before(int32_t in, int32_t out, int32_t k,
                              int32_t stride, int32_t dilation,
                              Padding padding) {
  if (padding != SAME) return 0;
  const int32_t total = (out - 1) * stride + (k - 1) * dilation + 1 - in;
  return total > 0 ? total / 2 : 0;
}

// EVENTS AND ERRORS
DECLARE_ERROR(OperatorIOSizeMismatchError);

// Operators do not go on the heap
class OperatorBase {
 protected:
  TensorMapInterface* _p_inputs;
  TensorMapInterface* _p_outputs;

 public:
  pTensor::string op_name;

 public:
  // OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs);
  // The preferred interface
  OperatorBase();
  OperatorBase(pTensor::string op_name);
  void set_name(pTensor::string _name);
  virtual ~OperatorBase();

  // Operators and models are static or stack objects, never heap objects.
  // Deleting operator new, and giving the class its own (no-op) operator
  // delete for the virtual destructor, keeps malloc/free out of the binary.
  static void* operator new(size_t) = delete;
  static void operator delete(void*) {}

  void eval();

 protected:
  OperatorBase(TensorMapInterface* inputs);
  void set_inputs(TensorMapInterface* inputs);
  void set_outputs(TensorMapInterface* outputs);

  friend class Context;
  virtual void compute() = 0;
};

template <size_t num_inputs>
class InPlaceOperatorInterface : public OperatorBase {
  protected:
    FixedTensorMap<num_inputs> inputs;
  
  public:
    InPlaceOperatorInterface() : OperatorBase() {}
    virtual ~InPlaceOperatorInterface() {}
  
    // This will throw compile time errors if users provide the wrong number of
    // inputs
    InPlaceOperatorInterface& set_inputs(FixedTensorMap<num_inputs>&& in) {
      inputs = in;
      OperatorBase::set_inputs(&inputs);
      return *this;
    }
  
  protected:
    virtual void compute() = 0;
};

template <size_t num_inputs, size_t num_outputs>
class OperatorInterface : public OperatorBase {
 protected:
  FixedTensorMap<num_inputs> inputs;
  FixedTensorMap<num_outputs> outputs;

 public:
  OperatorInterface() : OperatorBase() {}
  virtual ~OperatorInterface() {}

  // This will throw compile time errors if users provide the wrong number of
  // inputs
  OperatorInterface& set_inputs(FixedTensorMap<num_inputs>&& in) {
    inputs = in;
    OperatorBase::set_inputs(&inputs);
    return *this;
  }
  OperatorInterface& set_outputs(FixedTensorMap<num_outputs>&& out) {
    outputs = out;
    OperatorBase::set_outputs(&outputs);
    return *this;
  }

 protected:
  virtual void compute() = 0;
};

// Just a tag for Tensor access power
class FastOperator {
 public:
  size_t get_readable_block(const Tensor& t, const void*& buffer, uint16_t num_elems,
                            int linear_index);
  size_t get_writeable_block(Tensor& t, void*& buffer, uint16_t num_elems,
                             int linear_index);
};
}  // namespace pTensor
#endif