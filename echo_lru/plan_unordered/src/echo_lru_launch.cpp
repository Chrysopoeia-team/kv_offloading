// echo_lru_launch.cpp - host launchers of kernel 1 (decode step and init).
// Compiled as the kernel-1 translation unit together with echo_lru_kernel.h.

#include "echo_lru_launch.h"
#include "echo_lru_kernel.h"

void launch_echo_lru(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* freeSlots,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, uint8_t* fresh, int32_t* curSlots,
                     int32_t* missCount, int32_t* missTokens, int32_t* missSlots,
                     void* workspace, void* tiling, uint32_t blockNum, uint32_t blockDim,
                     void* stream) {
  ascend_kernel::echo_lru_kernel<<<blockNum, blockDim, 0, stream>>>(
      htd, dth, lru, freeSlots, avail, fifo, pos, spl, resetMask, fresh, curSlots,
      missCount, missTokens, missSlots, workspace,
      reinterpret_cast<ascend_kernel::EchoLruTiling*>(tiling));
}

void launch_echo_lru_init(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* freeSlots,
                          int32_t* avail, int32_t* fifo, uint8_t* fresh, uint8_t* initMask,
                          void* tiling, uint32_t blockNum, uint32_t blockDim, void* stream) {
  ascend_kernel::echo_lru_init_kernel<<<blockNum, blockDim, 0, stream>>>(
      htd, dth, lru, freeSlots, avail, fifo, fresh, initMask,
      reinterpret_cast<ascend_kernel::EchoLruTiling*>(tiling));
}
