#include "arenaAllocator.hpp"

namespace pTensor {

namespace {

// Align `p` up to `alignment` (a power of two) if the aligned block of `sz`
// bytes still fits in `space`. Mirrors std::align without pulling in <memory>.
void* align_up(size_t alignment, size_t sz, void*& p, size_t& space) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  const uintptr_t aligned = (addr + (alignment - 1)) & ~(uintptr_t)(alignment - 1);
  const size_t pad = aligned - addr;
  if (pad > space || sz > space - pad) return nullptr;
  space -= pad;
  p = reinterpret_cast<void*>(aligned);
  return p;
}

// Small, allocation-free insertion sort. The block table is tiny, so this is
// both smaller in flash and fast enough compared to std::sort.
template <typename T, typename Less>
void insertion_sort(T* first, T* last, Less less) {
  if (first == last) return;
  for (T* i = first + 1; i < last; ++i) {
    T v = *i;
    T* j = i;
    while (j > first && less(v, *(j - 1))) {
      *j = *(j - 1);
      --j;
    }
    *j = v;
  }
}

}  // namespace

DEFINE_EVENT(MetaHeaderNotFound);
DEFINE_EVENT(localCircularArenaAllocatorRebalancing);
DEFINE_EVENT(localCircularArenaAllocatorConstructed);
DEFINE_ERROR(InvalidBoundRegionState);
DEFINE_ERROR(InvalidAlignmentAllocation);
DEFINE_ERROR(MetaHeaderNotBound);
DEFINE_ERROR(ArenaBlockTableFullError);

localCircularArenaAllocatorBase::MetaHeader::MetaHeader()
    : meta_data(BLOCK_INACTIVE & BLOCK_ZERO_LENGTH),
      hndl(nullptr),
      _d(nullptr) {}
localCircularArenaAllocatorBase::MetaHeader::MetaHeader(uint32_t sz)
    : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr), _d(nullptr) {}
localCircularArenaAllocatorBase::MetaHeader::MetaHeader(uint32_t sz, uint8_t* d)
    : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr), _d(d) {}
void localCircularArenaAllocatorBase::MetaHeader::set_active() { meta_data |= BLOCK_ACTIVE; }
void localCircularArenaAllocatorBase::MetaHeader::set_inactive() { meta_data &= BLOCK_INACTIVE; }
void localCircularArenaAllocatorBase::MetaHeader::set_hndl(Handle* handle) { hndl = handle; }
void localCircularArenaAllocatorBase::MetaHeader::set_d(uint8_t* d) { _d = d; }
void localCircularArenaAllocatorBase::MetaHeader::set_len(uint32_t sz) {
  meta_data &= MSB_SET;  // Clear all size bits
  meta_data |= (BLOCK_LENGTH_MASK & sz);
}
uint32_t localCircularArenaAllocatorBase::MetaHeader::get_len() const { return meta_data & BLOCK_LENGTH_MASK; }
bool localCircularArenaAllocatorBase::MetaHeader::is_active() const { return (meta_data & MSB_SET) == BLOCK_ACTIVE; }
bool localCircularArenaAllocatorBase::MetaHeader::is_bound() const { return (hndl != nullptr); }
bool localCircularArenaAllocatorBase::MetaHeader::has_handle(Handle* target) const {
  return is_active() && (hndl == target);
}
bool localCircularArenaAllocatorBase::MetaHeader::is_used() const { return is_active() && (get_len() > 0); }

// Return the amount of free space at the tail
uint32_t localCircularArenaAllocatorBase::tail_capacity(){
  return (uint32_t) 0;
};

bool localCircularArenaAllocatorBase::push_header(const MetaHeader& hdr) {
  if (_num_headers >= _max_headers) return false;
  _headers[_num_headers++] = hdr;
  return true;
}
// This is just for reference
localCircularArenaAllocatorBase::MetaHeader& localCircularArenaAllocatorBase::_read_header(void* ptr) {
  static MetaHeader not_found;
  // First check if ptr in bounds
  if (ptr < begin() || ptr > end()) {
    // ERROR
    Context::get_default_context()->throwError(OutOfMemBoundsError());
  }
  for (auto hdr_i = headers_begin(); hdr_i != headers_end(); hdr_i++) {
    if (hdr_i->_d == ptr) return *hdr_i;
  }
  // ERROR
  Context::get_default_context()->notifyEvent(MetaHeaderNotFound());
  return not_found;
}

void* localCircularArenaAllocatorBase::attempt_to_reuse_inactive_region(size_t sz) {
  uint8_t* loc = nullptr;
   for (auto hdr_i = headers_begin(); hdr_i != headers_end(); hdr_i++) {
     if (!hdr_i->is_active() && hdr_i->get_len() >= sz) {
       MetaHeader& hdr = *hdr_i;
       // Handle alignment
       void* aligned_loc = (void*)hdr._d;
       size_t space_change = hdr.get_len();
       aligned_loc =
           align_up(alignof(uint8_t*), sz, aligned_loc, space_change);
       if(aligned_loc == nullptr){
         Context::get_default_context()->throwError(InvalidAlignmentAllocation());
       }
       hdr.set_active();
       // hdr.set_len(sz + hdr.get_len() - space_change);
       hdr.set_len(sz);
       hdr.set_hndl(nullptr);
       hdr.set_d((uint8_t*)aligned_loc);
       loc = (uint8_t*)aligned_loc;

       // Update capacity
       //capacity -= hdr.get_len();

       return (void*)loc;
     }
   }
   return nullptr;
}

inline uint8_t* localCircularArenaAllocatorBase::begin() const { return _buffer; }
inline const uint8_t* localCircularArenaAllocatorBase::end() const { return _buffer + size; }
inline size_t localCircularArenaAllocatorBase::_get_size() const { return size; }

void localCircularArenaAllocatorBase::_bind(void* ptr, Handle* hndl) {
  MetaHeader& hdr = _read_header(ptr);
  // Check if region is active
  if (!hdr.is_active()) {
    // ERROR
    Context::get_default_context()->throwError(InvalidBoundRegionState());
  }
  hdr.set_hndl(hndl);
}

 void localCircularArenaAllocatorBase::_unbind(void* ptr, Handle* hndl) {
  MetaHeader& hdr = _read_header(ptr);
  if (!hdr.is_active()) {
    // ERROR
    Context::get_default_context()->throwError(InvalidBoundRegionState());
  }
  // teehee
  update_hndl(hndl, nullptr);
  hdr.set_hndl(nullptr);
  //_bind(ptr, nullptr);
}

 bool localCircularArenaAllocatorBase::_is_bound(void* ptr, Handle* hndl) {
  MetaHeader hdr = _read_header(ptr);
  // Check if region is active
  if (!hdr.is_active()) {
    // ERROR
    Context::get_default_context()->throwError(MetaHeaderNotBound());
  }
  return hdr.is_bound() && (hdr.hndl == hndl);
  ;
}

 bool localCircularArenaAllocatorBase::_has_handle(Handle* hndl) {
  for (auto hdr_i = headers_begin(); hdr_i != headers_end(); hdr_i++) {
    if (hdr_i->has_handle(hndl)) return true;
  }
  return false;
}

 void* localCircularArenaAllocatorBase::_allocate(size_t sz) {
  uint8_t* loc = nullptr;
  // If make this capacity then have possibility of filling up
  if (sz > _get_size()) {
    // ERROR
    Context::get_default_context()->throwError(OutOfMemError());
    return nullptr;
  }
  // if(sz > ( end() - (cursor + sizeof(MetaHeader)))){
  if (sz > available()) {
    // Allocate at beginning
    // Rebalance to make it less likely to overwrite a region
    // Overwriting allocated regions is a valid operation as long as the
    // overwritten regions are invalidated
    rebalance();
    // If still dont have space, error out
    if (sz > available()) {
      Context::get_default_context()->throwError(OutOfMemError());
      return nullptr;
    }
  }

  // First check to see if we have space in a previously allocated area
  // TODO: if this region is smaller split it and add another header to the
  // table
  void* reallocated = attempt_to_reuse_inactive_region(sz);
  if(reallocated){
    return reallocated;
  }
  if (sz > (end() - reinterpret_cast<uint8_t*>(cursor))){
    rebalance();

  }
  // Otherwise allocate at the end. Make sure the block table has room first;
  // a rebalance drops every inactive block (and moves the cursor), so it must
  // happen before the new block's location is computed.
  if (_num_headers >= _max_headers) {
    rebalance();
    if (_num_headers >= _max_headers) {
      Context::get_default_context()->throwError(ArenaBlockTableFullError());
      return nullptr;
    }
  }
  if (sz > available()) {
    Context::get_default_context()->throwError(OutOfMemError());
    return nullptr;
  }
  MetaHeader hdr;
  // Handle alignment
  void* aligned_loc = (void*)cursor;
  size_t space_change = available();
  aligned_loc = align_up(alignof(uint8_t*), sz, aligned_loc, space_change);
  if(aligned_loc == nullptr){
    Context::get_default_context()->throwError(InvalidAlignmentAllocation());
    return nullptr;
  }
  hdr.set_active();
  // hdr.set_len(sz + available() - space_change);
  hdr.set_len(sz);
  hdr.set_hndl(nullptr);
  hdr.set_d((uint8_t*)aligned_loc);
  push_header(hdr);  // Space in the table was ensured above.
  loc = (uint8_t*)aligned_loc;
  //cursor += hdr.get_len() + available() - space_change;
  cursor = reinterpret_cast<uint8_t*>(aligned_loc);
  cursor += hdr.get_len();

  // Update capacity (accounts for any alignment padding as well)
  capacity = end() - cursor;

  return (void*)loc;
}

 void localCircularArenaAllocatorBase::_deallocate(void* ptr) {
  if (ptr) {
    MetaHeader& hdr = _read_header(ptr);
    if (hdr.is_bound()) {
      _unbind(ptr, hdr.hndl);
    }
    hdr.set_inactive();
    hdr.set_hndl(nullptr);  // cleanup
    //capacity += hdr.get_len();
    // Do not update the size of the header
  }
}

localCircularArenaAllocatorBase::localCircularArenaAllocatorBase(
    uint8_t* buffer, size_t size, MetaHeader* headers, size_t max_headers)
    : _buffer(buffer),
      size(size),
      capacity(size),
      cursor(buffer),
      _headers(headers),
      _num_headers(0),
      _max_headers(max_headers) {
  Context::get_default_context()->notifyEvent(
      localCircularArenaAllocatorConstructed());
  cursor = begin();
}
 localCircularArenaAllocatorBase::~localCircularArenaAllocatorBase() {
  
}

/** This implementation of rebalance shifts all allocated chunks to the end of
 * the buffer and inserts an inactive region at the start. note: cursor gets
 * moved to begin() note: unbound regions get wiped
 */
// TODO Check to make sure updated locations are still aligned
 bool localCircularArenaAllocatorBase::rebalance() {
  Context::get_default_context()->notifyEvent(
      localCircularArenaAllocatorRebalancing());
  // Clear all unbound entries, and compact the table so it only holds the
  // bound (still referenced) regions. This gives the allocator a cold start.
  size_t kept = 0;
  for (size_t i = 0; i < _num_headers; ++i) {
    if (_headers[i].is_bound() && _headers[i].is_active()) {
      _headers[kept++] = _headers[i];
    }
  }
  _num_headers = kept;

  // Sort by region so data can be shifted down without overlap.
  insertion_sort(headers_begin(), headers_end(),
                 [](const MetaHeader& a, const MetaHeader& b) {
                   return a._d < b._d;
                 });

  uint8_t tmp;
  cursor = begin();
  void* aligned_loc;
  //size_t space_change;

  for (auto hdr_i = headers_begin(); hdr_i != headers_end(); hdr_i++) {
    aligned_loc = (void*)cursor;
    //size_t space_change = available();
    size_t space_change = (hdr_i->_d - cursor) + hdr_i->get_len();
    aligned_loc = align_up(alignof(uint8_t*), hdr_i->get_len(), aligned_loc,
                             space_change);
    if(aligned_loc == nullptr){
      Context::get_default_context()->throwError(InvalidAlignmentAllocation());
    }

    // Shift the data
    // Regions only ever move towards the start of the buffer, so a forward
    // overlapping copy is safe.
    for (uint32_t i = 0; i < hdr_i->get_len(); i++) {
      tmp = hdr_i->_d[i];
      reinterpret_cast<uint8_t*>(aligned_loc)[i] = tmp;
    }

    // Update header
    // hdr_i->set_len(sz + available() - space_change);
    hdr_i->set_d((uint8_t*)aligned_loc);
    update_hndl(hdr_i->hndl, hdr_i->_d);
    //cursor += hdr_i->get_len() + available() - space_change;
    cursor = reinterpret_cast<uint8_t*>(aligned_loc);
    cursor += hdr_i->get_len();
  }
  capacity = end() - cursor;
  return true;
}

 size_t localCircularArenaAllocatorBase::available() { return capacity; }

 void localCircularArenaAllocatorBase::clear() {
  // TODO deallocate and invalidate all references
  // reset to default state
  memset(_buffer, 0, size);
  _num_headers = 0;
  cursor = begin();
  capacity = _get_size();
}

// Check to see if pointer exists in memory space and is valid
bool localCircularArenaAllocatorBase::contains(void* p) const {
  if (!((p >= begin()) && (p < end()))) {
    return false;
  }
  for (auto hdr_i = headers_begin(); hdr_i != headers_end(); hdr_i++) {
    if (hdr_i->_d == p) return hdr_i->is_used();
  }
  return false;
  // MetaHeader hdr = _read_header(p);
  // return hdr.is_used();
}

// Testing bits, attribute out later
uint32_t localCircularArenaAllocatorBase::internal_header_unit_size() const { return sizeof(MetaHeader); }



}  // namespace pTensor
