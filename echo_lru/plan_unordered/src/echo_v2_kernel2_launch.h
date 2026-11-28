#pragma once
// echo_v2_kernel2_launch.h - host launch interface of kernel 2.


#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

#ifdef __cplusplus
extern "C" {
#endif

void launch_echo_v2_kernel2(GM_ADDR lru, GM_ADDR dth, GM_ADDR htd, GM_ADDR free,
                            GM_ADDR avail, GM_ADDR fifo, GM_ADDR pos, GM_ADDR curSlots,
                            GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                            GM_ADDR workspace, GM_ADDR tiling,
                            uint32_t B, uint32_t topkDim, uint32_t rowsPerCore,
                            uint32_t blockNum, void* stream);

#ifdef __cplusplus
}
#endif
