// echo_lru_launch.cpp - host launchers of kernel 1 (decode step and init).
// Compiled as the kernel-1 translation unit together with echo_lru_kernel.h.

#include "echo_lru_kernel.h"
#include "echo_lru_launch.h"

void launch_echo_lru(GM_ADDR htd, GM_ADDR dth, GM_ADDR lru, GM_ADDR freeSlots,
                     GM_ADDR avail, GM_ADDR fifo, GM_ADDR pos, GM_ADDR spl,
                     GM_ADDR resetMask, GM_ADDR fresh, GM_ADDR curSlots,
                     GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                     GM_ADDR workspace, GM_ADDR tiling,
                     uint32_t B, uint32_t topkDim, uint32_t specPositions,
                     uint32_t blockNum, void* stream) {
  echo_lru_kernel<<<blockNum,
                    ascend_kernel::echo_ws_elems(topkDim, B, specPositions) * sizeof(int32_t),
                    stream>>>(
      htd, dth, lru, freeSlots, avail, fifo, pos, spl, resetMask, fresh, curSlots,
      missCount, missTokens, missSlots, workspace, tiling);
}

void launch_echo_lru_init(GM_ADDR htd, GM_ADDR dth, GM_ADDR lru, GM_ADDR freeSlots,
                          GM_ADDR avail, GM_ADDR fifo, GM_ADDR fresh, GM_ADDR initMask,
                          GM_ADDR tiling, uint32_t blockNum, void* stream) {
  echo_lru_init_kernel<<<blockNum, ascend_kernel::ECHO_WS_INIT_UB_BYTES, stream>>>(
      htd, dth, lru, freeSlots, avail, fifo, fresh, initMask, tiling);
}
