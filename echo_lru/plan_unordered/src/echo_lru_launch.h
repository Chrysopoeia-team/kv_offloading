#pragma once
// echo_lru_launch.h - host launch interface of kernel 1 and the workspace
// layout shared by both kernels.

#include <cstdint>
#include "echo_lru_tiling.h"

void launch_echo_lru(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* freeSlots,
                     int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                     uint8_t* resetMask, uint8_t* fresh, int32_t* curSlots,
                     int32_t* missCount, int32_t* missTokens, int32_t* missSlots,
                     void* workspace, void* tiling, uint32_t blockNum, uint32_t blockDim,
                     void* stream);

void launch_echo_lru_init(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* freeSlots,
                          int32_t* avail, int32_t* fifo, uint8_t* fresh, uint8_t* initMask,
                          void* tiling, uint32_t blockNum, uint32_t blockDim, void* stream);

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
  uint64_t base = (l.endOff + 1ULL) & ~1ULL;
  uint64_t perBlock = 0;
#ifdef ECHO_PERF_STEP
  perBlock += ascend_kernel::ECHO_PERF_NUM_STEPS;
#endif
  return base + uint64_t(blocks) * perBlock;
}
