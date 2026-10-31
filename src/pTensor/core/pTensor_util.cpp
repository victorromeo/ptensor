#include "pTensor_util.hpp"
#if !(defined(TARGET_MBED) || defined(ARDUINO))
//#include <dirent.h>
//#include <sys/stat.h>
#endif
#include <cstdlib>

void return_error(int ret_val) {
  if (ret_val) {
    pTensor_printf(" [**Failure**] %d\r\n", ret_val);
    pTensor_printf("Exiting...\r\n");
    pTensor_flush();
    ptensor_exit();
  } else {
    pTensor_printf("  [DONE]\r\n");
  }
}
/*
void printVector(std::vector<uint32_t> vec) {
  printf("vector: \r\n");
  for (uint32_t i : vec) {
    printf("%d ", (unsigned int)i);
  }

  printf("\r\n");
}
*/

uint16_t ntoh16(uint16_t val) {
  uint16_t ret = 0;

  ret |= val >> 8;
  ret |= val << 8;

  return ret;
}

uint32_t ntoh32(uint32_t val) {
  const uint32_t mask = 0b11111111;
  uint32_t ret = 0;

  ret |= val >> 24;
  ret |= (val & (mask << 16)) >> 8;
  ret |= (val & (mask << 8)) << 8;
  ret |= val << 24;

  return ret;
}

#ifdef ARDUINO
void int_env() {}
#else
void init_env() {
//  int status;
//  DIR* dir = opendir("/fs/tmp");
//  if (dir) {
//    closedir(dir);
//    return;
//  } else {
//    status = mkdir("/fs/tmp", 0777);
//    if (status == -1) {
//      ERR_EXIT("env setting failed");
//    }
//  }
}
#endif

// On bare-metal targets there is nothing to exit to, and calling exit() would
// drag in the C runtime's atexit/stdio teardown, so just halt.
void ptensor_exit(void) {
#if !defined(PTENSOR_BARE_METAL) && !defined(__EMSCRIPTEN__)
  exit(-1);
#endif
  while (true) {
  }
}
