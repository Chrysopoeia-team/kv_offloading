// echo_lru_tiling.h - compile-time limits and the tiling struct shared by the
// kernels and the host operator. Plain C++ so that both bisheng and the host
// compiler can include it.

#ifndef ECHO_LRU_TILING_H
#define ECHO_LRU_TILING_H

#include <cstdint>

namespace ascend_kernel {

#ifdef ECHO_LRU_THREAD_NUM_OVERRIDE
constexpr uint32_t ECHO_LRU_THREAD_NUM = ECHO_LRU_THREAD_NUM_OVERRIDE;
#else
constexpr uint32_t ECHO_LRU_THREAD_NUM = 1024;
#endif
static_assert(ECHO_LRU_THREAD_NUM % 32u == 0u, "ECHO_LRU_THREAD_NUM must be a multiple of warp size 32");
static_assert(ECHO_LRU_THREAD_NUM >= 64u, "ECHO_LRU_THREAD_NUM must cover plugin blockDim floor");
static_assert(ECHO_LRU_THREAD_NUM <= 1024u, "ECHO_LRU_THREAD_NUM exceeds AIV launch bounds");

// Upper bound of the per-request cache capacity (slots).
constexpr uint32_t ARGTOPK_B_MAX = 8192;

constexpr uint32_t ARGTOPK_NPAD_MAX = 128;
constexpr uint32_t ARGTOPK_GM_NPAD_MAX = 8192;
constexpr uint32_t ARGTOPK_UB_NPAD_MAX = 8192;

constexpr uint32_t ARGTOPK_UB_ELEMS = 4u * ARGTOPK_UB_NPAD_MAX;
constexpr uint32_t ARGTOPK_GROUP_THREADS = 64;

// lru rows are padded to capacity + 8 entries.
constexpr uint32_t ECHO_LRU_ROW_PAD = 8;
#define ECHO_LRU_ROW_PITCH(B_) ((B_) + (ascend_kernel::ECHO_LRU_ROW_PAD))

constexpr uint32_t ECHO_PERF_NUM_STEPS = 24;

constexpr uint32_t ECHO_PERF_CLS_SUB_N = 16;

#ifdef ECHO_V2_DEBUG_CUTOFF
constexpr uint32_t ECHO_V2_DEBUG_CUTOFF_V = ECHO_V2_DEBUG_CUTOFF;
#else
constexpr uint32_t ECHO_V2_DEBUG_CUTOFF_V = 0;
#endif
// Per-call launch parameters, read by the kernels from GM.
struct EchoLruTiling {

    uint32_t blockNum;
    uint32_t rowsPerCore;
    uint32_t tailRows;

    uint32_t numRequests;
    uint32_t numTokens;

    uint32_t topkBufferB;
    uint32_t freeStackSize;
    uint32_t maxModelLenP1;
    uint32_t topkDim;
    uint32_t effTopk;
    uint32_t specPositions;

    uint32_t blockSize;
    uint32_t blockTableCols;
    uint32_t kDim;
    uint32_t vDim;
    uint32_t kvHeadNum;
    uint32_t specEnabled;

    uint32_t argtopkDebugRow;
    uint32_t argtopkGroupThreads;
};

}

#endif
