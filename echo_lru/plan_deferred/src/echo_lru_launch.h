#pragma once
// echo_lru_launch.h - host launch interface of kernel 1 and the workspace
// layout shared by both kernels.


#include <cstdint>

#include "echo_lru_tiling.h"

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Offsets of the workspace regions (int32 elements) handed from kernel 1 to
// kernel 2.
struct EchoWorkspaceLayout {
  uint64_t freeIdxOff;
  uint64_t masksOff;
  uint64_t countsOff;
  uint64_t handoffOff;
  uint64_t endOff;
};

inline EchoWorkspaceLayout EchoWorkspaceLayoutOf(uint32_t requests, uint32_t tokens,
                                                 uint32_t topk, uint32_t capacity, uint32_t blocks) {
  const uint64_t R = uint64_t(requests);
  const uint64_t Tk = uint64_t(tokens);
  const uint64_t B = uint64_t(capacity);
  EchoWorkspaceLayout l{};
  uint64_t o = 0;
  o += R * (B + 1ULL);
  o += R;
  o += R;
  o += R;
  l.freeIdxOff = o;
  o += Tk * uint64_t(topk);
  o += R;
  o += Tk * uint64_t(topk);
  o += Tk * uint64_t(topk);
  o = (o + 7ULL) & ~7ULL;
  l.masksOff = o;
  o += R * (3ULL * ((B + 31ULL) / 32ULL));
  l.countsOff = o;  o += R * 8ULL;
  l.handoffOff = o; o += uint64_t(blocks) * 2ULL;
  l.endOff = o;
  return l;
}

// Workspace size required by echo_lru, in int32 elements.
inline uint64_t EchoWorkspaceElements(uint32_t requests, uint32_t tokens,
                                      uint32_t topk, uint32_t capacity, uint32_t blocks) {
  const EchoWorkspaceLayout l = EchoWorkspaceLayoutOf(requests, tokens, topk, capacity, blocks);
  return (l.endOff + 1ULL) & ~1ULL;
}

#ifdef __cplusplus
extern "C" {
#endif

void launch_echo_lru(GM_ADDR htd, GM_ADDR dth, GM_ADDR lru, GM_ADDR freeSlots,
                     GM_ADDR avail, GM_ADDR fifo, GM_ADDR pos, GM_ADDR spl,
                     GM_ADDR resetMask, GM_ADDR fresh, GM_ADDR curSlots,
                     GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                     GM_ADDR workspace, GM_ADDR tiling,
                     uint32_t B, uint32_t topkDim, uint32_t specPositions,
                     uint32_t blockNum, void* stream);

void launch_echo_lru_init(GM_ADDR htd, GM_ADDR dth, GM_ADDR lru, GM_ADDR freeSlots,
                          GM_ADDR avail, GM_ADDR fifo, GM_ADDR fresh, GM_ADDR initMask,
                          GM_ADDR tiling, uint32_t blockNum, void* stream);

#ifdef __cplusplus
}
#endif
