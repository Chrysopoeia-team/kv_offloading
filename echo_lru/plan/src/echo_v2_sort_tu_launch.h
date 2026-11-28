#pragma once
// echo_v2_sort_tu_launch.h - declarations of a sort kernel from an earlier design.
//
// echo_v2_kernel2_launch.h includes this header, so it is kept unchanged. The
// kernel is no longer launched by the operator and its translation unit is not
// part of the library.


#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

#ifdef __cplusplus
extern "C" {
#endif

void launch_echo_v2_sort(GM_ADDR dth, GM_ADDR resetMask, GM_ADDR avail,
                         GM_ADDR workspace, GM_ADDR tiling,
                         uint32_t B, uint32_t blockNum, void* stream);

#ifdef __cplusplus
}
#endif
