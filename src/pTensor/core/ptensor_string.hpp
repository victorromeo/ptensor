#ifndef PTENSOR_STRING
#define PTENSOR_STRING
//#include <string.h>
#include <cstdint>
#include <cstring>

namespace pTensor {
class string {
 private:
  uint32_t value;
  const char* cstr = NULL;

  uint32_t hash(const char* c) {
    // Unsigned arithmetic: wrap-around is well defined (signed overflow is UB).
    uint32_t v = 7;
    for (; *c; ++c) {
      v = v * 31u + static_cast<uint32_t>(static_cast<unsigned char>(*c));
    }
    return v;
  }

 public:
  string(const char* that) {
    value = hash(that);
    cstr = that;
  }
  string() { value = 0; }
  string(uint32_t val) : value(val), cstr(NULL) {}
  // bool operator < (const string& that){ return this->value < that.value; }
  // bool operator == (const string& that){ return this->value == that.value; }
  bool operator<(const string& that) const { return this->value < that.value; }
  bool operator==(const string& that) const {
    return this->value == that.value;
  }
  bool operator!=(const string& that) const {
    return this->value != that.value;
  }

  uint32_t get_value() const { return value; }
  const char* c_str() const { return cstr; }
};

}  // namespace pTensor

// namespace std {
// template <>
// struct hash<pTensor::string> {
//  typedef pTensor::string argument_type;
//  typedef std::size_t result_type;
//
//  result_type operator()(argument_type const& s) const noexcept {
//    return s.get_value();
//  }
//};
//}  // namespace std

#endif
