// echo_lru_kernel.h - kernel 1 of the Echo LRU plan operator: lookup and classification.
//
// One block handles a contiguous range of request rows. All P prediction positions
// of a request share one LRU pool and are processed by the same block. For every
// row the kernel:
//   1. applies reset, fresh and speculative-suffix handling to the persistent state,
//   2. turns the P sorted TopK rows into one sorted list of unique tokens
//      (P = 1: direct build; P > 1: pack, hardware merge sort, dedup),
//   3. walks the LRU order of the resident slots and looks each slot's token up in
//      that list, which yields the hit slot of every query,
//   4. writes current_slots, the ordered miss list and the victim / survivor / hit
//      masks and counts that kernel 2 consumes.
//
// The file is a mixed translation unit: a scalar/vector shell (TPipe/TBuf workset,
// MrgSort, DataCopy) that launches SIMT frames with asc_vf_call. Callees are
// __simt_callee__ functions, frames are __simt_vf__ functions.
//
// Reading order: echo_lru_kernel -> EchoWsFrontFrame -> EchoWsVfFrame (P = 1) and
// EchoWsPackFrame / EchoWsJointFrame (P > 1) -> WsBuildDirectRow or WsDedupVf ->
// WsScanRowUb / WsScanRow -> WsFinalizeFlatRow / WsFinalizeVf -> OpEmitMasksFused
// or OpClassifyMaskEmit2. echo_lru_init_kernel builds the empty-pool state of
// new request rows outside the decode step.

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_warp_functions.h"
#include "echo_lru_tiling.h"
#include "echo_lru_launch.h"
#ifdef ECHO_WS_PHASE
#include "utils/debug/asc_time.h"
#endif

#ifndef KERNEL_TASK_TYPE_DEFAULT
// Fallback for toolchains that do not provide the task-type registration macro.
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace ascend_kernel {

constexpr int32_t ECHO_WS_I32_MAX = 2147483647;

#ifdef ECHO_WS_VF_THREADS_OVERRIDE
constexpr uint32_t ECHO_WS_VF_THREADS = ECHO_WS_VF_THREADS_OVERRIDE;
#else
constexpr uint32_t ECHO_WS_VF_THREADS = 512u;
#endif

constexpr uint32_t ECHO_WS_SCAN_BATCH = 8u;

#define ECHO_WS_ALIGN8(x_) (((x_) + 7u) & ~7u)
constexpr uint32_t ECHO_WS_SCAN_ELEMS = 128u;
constexpr uint32_t ECHO_WS_RB_ELEMS = 2u * ECHO_WS_VF_THREADS + 64u;
#define ECHO_WS_A_WORDS(topk_, P_) (2u * (P_) * (topk_))

#define ECHO_WS_B_WORDS(topk_, B_, P_) \
    (ECHO_WS_ALIGN8(2u * (P_) * (topk_)) + ECHO_WS_ALIGN8((B_) + 1u))
#define ECHO_WS_A_OFF (ECHO_WS_SCAN_ELEMS + ECHO_WS_RB_ELEMS)
#define ECHO_WS_RB_OFF (ECHO_WS_SCAN_ELEMS)

#define ECHO_WS_FMASK_OFF ECHO_WS_ALIGN8(257u)
#define ECHO_WS_B_OFF(topk_, P_) (ECHO_WS_A_OFF + ECHO_WS_A_WORDS(topk_, P_))

#define ECHO_WS_UNIQUE_OFF(topk_, P_) (ECHO_WS_A_OFF + ECHO_WS_ALIGN8((P_) * (topk_)))

#define ECHO_WS_OWNER_OFF(topk_, P_) (((P_) == 1u) ? (ECHO_WS_B_OFF(topk_, P_)) : (ECHO_WS_A_OFF))

#define ECHO_WS_FLAGS_OFF(topk_, B_, P_) (ECHO_WS_B_OFF(topk_, P_) + ECHO_WS_ALIGN8(2u * (P_) * (topk_)))

#define ECHO_WS_WS_ELEMS(topk_, B_, P_) \
    (ECHO_WS_ALIGN8(ECHO_WS_SCAN_ELEMS + ECHO_WS_RB_ELEMS + ECHO_WS_A_WORDS(topk_, P_) + ECHO_WS_B_WORDS(topk_, B_, P_)))

#define ECHO_WS_DTH_SLACK(topk_, B_, P_) \
    (((49152u - 16u) > ECHO_WS_WS_ELEMS(topk_, B_, P_)) \
         ? ((49152u - 16u) - ECHO_WS_WS_ELEMS(topk_, B_, P_)) \
         : 0u)
#define ECHO_WS_DTH_CAP(topk_, B_, P_) \
    ((P_) == 1u \
         ? (ECHO_WS_DTH_SLACK(topk_, B_, P_) / ((B_) + 1u) > 0u \
                ? ECHO_WS_DTH_SLACK(topk_, B_, P_) / ((B_) + 1u) \
                : 1u) \
         : 0u)

#define ECHO_WS_DTH_WORDS(topk_, B_, P_) \
    ((ECHO_WS_DTH_CAP(topk_, B_, P_)) > 0u \
         ? (ECHO_WS_ALIGN8(ECHO_WS_DTH_CAP(topk_, B_, P_) * ((B_) + 1u)) + 8u) \
         : 0u)
#define ECHO_WS_ELEMS(topk_, B_, P_) \
    (ECHO_WS_WS_ELEMS(topk_, B_, P_) + ECHO_WS_DTH_WORDS(topk_, B_, P_))

// Workset size of kernel 1 in int32 elements. Single source for the device
// InitBuffer call and for the dynamic UB size requested by the launcher.
constexpr uint32_t echo_ws_elems(uint32_t topk_, uint32_t B_, uint32_t P_) {
    return ECHO_WS_ELEMS(topk_, B_, P_);
}

// Block-wide exclusive scan of one value per thread (warp shuffle scan plus a
// warp-total table). Returns the exclusive prefix of the calling thread.
__simt_callee__ __aicore__ inline int32_t WsExclusiveScan(
    int32_t value, __ubuf__ int32_t* warpTotals) {
    uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    uint32_t warp = static_cast<uint32_t>(threadIdx.x) >> 5;
    uint32_t warpCount = ECHO_WS_VF_THREADS >> 5;
    int32_t inclusive = value;
    for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
        int32_t prior = asc_shfl_up(inclusive, offset, 32u);
        if (lane >= offset) inclusive += prior;
    }
    if (lane == 31u) warpTotals[warp] = inclusive;
    asc_syncthreads();
    int32_t base = 0;
    for (uint32_t w = 0u; w < warp; ++w) base += warpTotals[w];
    return base + inclusive - value;
}

// Block barrier followed by a D-cache flush, so that GM writes of this block
// are visible to the reads that follow.
__simt_callee__ __aicore__ inline void WsSyncGrid(__gm__ int32_t* wsBase) {
    asc_syncthreads();
    if (threadIdx.x == 0u) {
        asc_dcci_entire(reinterpret_cast<__gm__ void*>(wsBase));
    }
    asc_syncthreads();
}

#ifdef ECHO_WS_PHASE

// Phase timestamp written into the workspace tail; only active in the
// phase-instrumented build (ECHO_WS_PHASE).
__aicore__ inline void WsPhaseStamp(__gm__ int32_t* p, uint32_t k) {
    p[k] = static_cast<int32_t>(clock());
}
#endif

// Reset, htd: for every row flagged in reset_mask, htd becomes
// (0, INT32_MAX, INT32_MAX, ...).
__simt_callee__ __aicore__ inline void OpResetRows2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t maxModelLenP1,
    __gm__ uint8_t* resetMaskGm, __gm__ int32_t* htdGm) {
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * maxModelLenP1;
        for (uint32_t col = threadIdx.x; col < maxModelLenP1; col += ECHO_WS_VF_THREADS) {
            htdGm[rowOff + col] = (col == 0) ? 0 : ECHO_WS_I32_MAX;
        }
    }
}

// Reset, dth and lru: dth is set to INT32_MAX and the lru row is zeroed.
__simt_callee__ __aicore__ inline void OpResetDthLru2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B,
    __gm__ uint8_t* resetMaskGm, __gm__ int32_t* dthGm, __gm__ int32_t* lruGm) {

    uint32_t dthCols = B + 1u;
    uint32_t lruPitch = ECHO_LRU_ROW_PITCH(B);
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        if (resetMaskGm[row] == 0) continue;
        uint32_t dthRowOff = row * dthCols;
        for (uint32_t col = threadIdx.x; col < dthCols; col += ECHO_WS_VF_THREADS) {
            dthGm[dthRowOff + col] = ECHO_WS_I32_MAX;
        }
        uint32_t lruRowOff = row * lruPitch;
        for (uint32_t col = threadIdx.x; col < lruPitch; col += ECHO_WS_VF_THREADS) {
            lruGm[lruRowOff + col] = 0;
        }
    }
}

// Reset, free stack: rebuilt as [B, B-1, ..., 1] followed by zeros.
__simt_callee__ __aicore__ inline void OpResetFree2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t freeStackSize,
    __gm__ uint8_t* resetMaskGm, __gm__ int32_t* freeGm) {
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * freeStackSize;
        for (uint32_t col = threadIdx.x; col < freeStackSize; col += ECHO_WS_VF_THREADS) {
            freeGm[rowOff + col] = (col < B) ? (int32_t)(B - col) : 0;
        }
    }
}

// Reset, counters: avail = B and fifo = 1 for every flagged row.
__simt_callee__ __aicore__ inline void OpResetAvailFifo2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B,
    __gm__ uint8_t* resetMaskGm, __gm__ int32_t* availGm, __gm__ int32_t* fifoGm) {
    if (threadIdx.x == 0u) {
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            if (resetMaskGm[row] != 0) {
                availGm[row] = (int32_t)B;
                fifoGm[row] = 1;
            }
        }
    }
}

// Prologue helper: writes 0 into one int32 per row.
__simt_callee__ __aicore__ inline void OpFillZero2(
    uint32_t rowStart, uint32_t rowEnd, __gm__ int32_t* outGm) {
    if (threadIdx.x == 0u) {
        for (uint32_t row = rowStart; row < rowEnd; ++row) outGm[row] = 0;
    }
}

// Clears the fresh flag of each row once the first-step load has consumed it.
__simt_callee__ __aicore__ inline void OpFreshClear2(
    uint32_t rowStart, uint32_t rowEnd, __gm__ uint8_t* freshGm) {

    if (threadIdx.x == 0u) {
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            if (freshGm[row] != 0u) freshGm[row] = 0;
        }
    }
}

// Zeroes the mask and count scratch of each row (the kernel-2 input surface
// of rows that skip the scan).
__simt_callee__ __aicore__ inline void OpColdScratchZero2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t maskRowElems,
    __gm__ int32_t* masksGm, __gm__ int32_t* countsGm) {

    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        uint32_t mOff = row * maskRowElems;
        for (uint32_t i = threadIdx.x; i < maskRowElems; i += ECHO_WS_VF_THREADS) {
            masksGm[mOff + i] = 0;
        }
        uint32_t cOff = row * 8u;
        for (uint32_t i = threadIdx.x; i < 8u; i += ECHO_WS_VF_THREADS) {
            countsGm[cOff + i] = 0;
        }
    }
}

// Speculative-suffix invalidation, step 1: every resident token at or after
// spl[row] is dropped from dth and htd and flagged in the cumsum scratch.
__simt_callee__ __aicore__ inline void OpInvalidateSpecSuffix2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t maxModelLenP1,
    __gm__ int32_t* splGm, __gm__ int32_t* dthGm, __gm__ int32_t* htdGm,
    __gm__ int32_t* wsCumsumGm) {
    uint32_t cols = B + 1u;
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        int32_t spl = splGm[row];
        uint32_t rowOff = row * cols;
        uint32_t htdRowOff = row * maxModelLenP1;
        for (uint32_t slot = threadIdx.x; slot < cols; slot += ECHO_WS_VF_THREADS) {
            int32_t tok = dthGm[rowOff + slot];
            bool spec = (tok >= spl) && (tok != ECHO_WS_I32_MAX);
            wsCumsumGm[rowOff + slot] = spec ? 1 : 0;
            if (spec) {
                dthGm[rowOff + slot] = ECHO_WS_I32_MAX;
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                htdGm[htdRowOff + tokSafe] = ECHO_WS_I32_MAX;
            }
        }
    }
}

// Speculative-suffix invalidation, step 2: prefix sum of the flags; the row
// total is stored as the push count.
__simt_callee__ __aicore__ inline void OpSpecCumsum2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B,
    __gm__ int32_t* wsCumsumGm, __gm__ int32_t* pushCountGm) {
    if (threadIdx.x == 0u) {
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            int32_t acc = 0;
            for (uint32_t slot = 0; slot <= B; ++slot) {
                int32_t m = wsCumsumGm[row * (B + 1u) + slot];
                acc += m;
                wsCumsumGm[row * (B + 1u) + slot] = acc;
            }
            pushCountGm[row] = acc;
        }
    }
}

// Speculative-suffix invalidation, step 3: push the freed slots onto the
// free stack.
__simt_callee__ __aicore__ inline void OpSpecPushFree2(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t freeStackSize,
    __gm__ int32_t* wsCumsumGm, __gm__ int32_t* availGm, __gm__ int32_t* freeGm) {
    uint32_t cols = B + 1u;
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        int32_t avail = availGm[row];
        uint32_t rowOff = row * cols;
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t slot = threadIdx.x; slot < cols; slot += ECHO_WS_VF_THREADS) {

            int32_t cum = wsCumsumGm[rowOff + slot];
            int32_t prevCum = (slot == 0) ? 0 : wsCumsumGm[rowOff + slot - 1];
            if (cum - prevCum == 1) {
                int32_t writeIdx = avail + cum - 1;
                if (writeIdx < 0) writeIdx = 0;
                if (writeIdx >= (int32_t)freeStackSize) writeIdx = (int32_t)freeStackSize - 1;
                freeGm[freeRowOff + writeIdx] = (int32_t)slot;
            }
        }
    }
}

// Speculative-suffix invalidation, step 4: avail grows by the push count.
__simt_callee__ __aicore__ inline void OpSpecAvailAdd2(
    uint32_t rowStart, uint32_t rowEnd,
    __gm__ int32_t* availGm, __gm__ int32_t* pushCountGm) {
    if (threadIdx.x == 0u) {
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            availGm[row] = availGm[row] + pushCountGm[row];
        }
    }
}

// First-step load of a fresh request row: every query token is admitted and
// the pool state is built directly.
__simt_callee__ __aicore__ inline void FreshLoadRow(
    uint32_t row, uint32_t B, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    uint32_t freeStackSize,
    __gm__ int32_t* availGm, __gm__ int32_t* fifoGm, __gm__ uint8_t* freshGm,
    __gm__ int32_t* freeGm, __gm__ int32_t* posGm, __gm__ int32_t* curSlotsGm,
    __gm__ int32_t* missCountGm, __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* htdGm, __gm__ int32_t* dthGm, __gm__ int32_t* lruGm) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    int32_t availOld = availGm[row];
    uint32_t rowOff = row * topkDim;
    uint32_t htdRowOff = row * maxModelLenP1;
    uint32_t dthRowOff = row * (B + 1u);
    uint32_t lruRowOff = row * ECHO_LRU_ROW_PITCH(B);
    uint32_t freeRowOff = row * freeStackSize;
    for (uint32_t j = t; j < topkDim; j += ECHO_WS_VF_THREADS) {
        if (j >= effTopk) {

            curSlotsGm[rowOff + j] = 0;
            missTokensGm[rowOff + j] = -1;
            missSlotsGm[rowOff + j] = -1;
            continue;
        }
        int32_t readIdx = availOld - (int32_t)effTopk + (int32_t)j;
        if (readIdx < 0) readIdx = 0;
        if (readIdx >= (int32_t)freeStackSize) readIdx = (int32_t)freeStackSize - 1;
        int32_t aslot = freeGm[freeRowOff + readIdx];
        int32_t mpos = posGm[rowOff + j];
        int32_t mposSafe = mpos < 0 ? 0 : (mpos > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : mpos);
        curSlotsGm[rowOff + j] = aslot;
        missTokensGm[rowOff + j] = mpos;
        missSlotsGm[rowOff + j] = aslot;
        htdGm[htdRowOff + mposSafe] = aslot;
        int32_t s = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);
        dthGm[dthRowOff + s] = mpos;
        int32_t ac = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);
        lruGm[lruRowOff + j] = ac;
    }
    asc_syncthreads();
    if (t == 0u) {
        availGm[row] = availOld - (int32_t)effTopk;
        fifoGm[row] = fifoGm[row] + 2;
        freshGm[row] = 0;
        missCountGm[row] = (int32_t)effTopk;
    }
}

// P > 1, build step 1: packs the valid prefix of each TopK row as
// (-token, flat id) records into the merge area of the workset.
__simt_callee__ __aicore__ inline void WsPackRow(
    uint32_t row, uint32_t topkDim, uint32_t effTopk, uint32_t P,
    uint32_t flatBase, uint32_t posRowStride, uint32_t maxModelLenP1,
    __gm__ int32_t* posGm, __gm__ int32_t* curSlotsGm, __ubuf__ int32_t* recUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t rowOff = row * posRowStride;
    for (uint32_t p = 0u; p < P; ++p) {

        __ubuf__ int32_t* run = recUb + 2u * p * topkDim;
        __ubuf__ float* runF = reinterpret_cast<__ubuf__ float*>(run);
        for (uint32_t i = t; i < effTopk; i += ECHO_WS_VF_THREADS) {
            uint32_t j = flatBase + p * topkDim + i;
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            runF[2u * i] = -static_cast<float>(tokSafe);
            run[2u * i + 1u] = (int32_t)j;
        }

        for (uint32_t i = effTopk + t; i < topkDim; i += ECHO_WS_VF_THREADS) {
            curSlotsGm[rowOff + flatBase + p * topkDim + i] = 0;
        }
    }
}

// P > 1, dedup (SIMT form): reads the merged records in (token, flat) order,
// marks the group heads and writes the unique token list and the flat-id stash.
__simt_callee__ __aicore__ inline uint32_t WsDedupRow(
    uint32_t row, uint32_t topkDim, uint32_t effTopk, uint32_t P,
    __gm__ int32_t* freeIdxGm, __ubuf__ int32_t* recUb, __ubuf__ int32_t* mapUb,
    __ubuf__ int32_t* uniqueUb, __ubuf__ int32_t* scanUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t n = P * effTopk;
    uint32_t W = (n + ECHO_WS_VF_THREADS - 1u) / ECHO_WS_VF_THREADS;
    uint32_t iLo = t * W;
    uint32_t iHi = (iLo + W < n) ? (iLo + W) : n;
    __ubuf__ float* recF = reinterpret_cast<__ubuf__ float*>(recUb);

    int32_t cnt = 0;
    for (uint32_t i = iLo; i < iHi; ++i) {
        int32_t tok = static_cast<int32_t>(-recF[2u * i]);
        int32_t prev = (i == 0u) ? -1 : static_cast<int32_t>(-recF[2u * (i - 1u)]);
        if (tok != prev) cnt += 1;
    }

    int32_t excl = WsExclusiveScan(cnt, scanUb);

    uint32_t rowOff = row * (P * topkDim);
    __gm__ int32_t* stashGm = freeIdxGm + rowOff;
    uint32_t uid = static_cast<uint32_t>(excl);
    for (uint32_t i = iLo; i < iHi; ++i) {
        int32_t tok = static_cast<int32_t>(-recF[2u * i]);
        int32_t prev = (i == 0u) ? -1 : static_cast<int32_t>(-recF[2u * (i - 1u)]);
        int32_t flat = recUb[2u * i + 1u];
        if (tok != prev) {
            uniqueUb[uid] = tok;
            uid += 1;
        }
        mapUb[static_cast<uint32_t>(flat)] = (int32_t)(uid - 1u);
        stashGm[i] = flat;
    }

    uint32_t warpCount = ECHO_WS_VF_THREADS >> 5;
    int32_t grand = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) grand += scanUb[w];
    return (uint32_t)(grand > 0 ? grand : 0);
}

// P > 1, dedup (SIMD form): vector version of WsDedupRow.
__simd_vf__ inline void WsDedupVf(
    __ubuf__ int32_t* recUb, __ubuf__ int32_t* uniqueUb,
    __ubuf__ int32_t* ctxUb, uint32_t n) {
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, zero, one, even, odd, previousIndex, lastIndex;
    Reg::RegTensor<uint32_t> flags, prefix, carry, uid, payload, packed;
    Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
    Reg::RegTensor<int32_t> token;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg first, head, valid;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(carry, uint32_t{0});
    Reg::Duplicate(lastScore, 1.0f);
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Adds(previousIndex, lane, uint32_t(-1), full);
    Reg::Select(previousIndex, zero, previousIndex, first);
    static constexpr Reg::CastTrait toInt = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    __ubuf__ uint32_t* recU = reinterpret_cast<__ubuf__ uint32_t*>(recUb);
    __ubuf__ float* recF = reinterpret_cast<__ubuf__ float*>(recUb);
    for (uint32_t base = 0; base < n; base += 64U) {
        uint32_t remaining = n - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);
        const uint32_t count = n - base < 64U ? n - base : 64U;
        Reg::Adds(even, lane, base, full);
        Reg::ShiftLefts(even, even, int16_t{1}, full);
        Reg::Adds(odd, even, uint32_t{1}, full);
        Reg::Duplicate(score, 0.0f);
        Reg::Gather<float, float, uint32_t>(score, recF, even, valid);
        Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
        Reg::Select(previous, lastScore, shiftScore, first);
        Reg::Compare<float, CMPMODE::NE>(head, score, previous, valid);
        Reg::Select(flags, one, zero, head);
        Reg::Select(flags, flags, zero, valid);
        Reg::Compares<uint32_t, CMPMODE::NE>(head, flags, uint32_t{0}, full);
        Reg::Unsqueeze<uint32_t>(prefix, head);
        Reg::Add(prefix, prefix, flags, full);
        Reg::Add(prefix, prefix, carry, full);
        Reg::Adds(uid, prefix, uint32_t(-1), full);
        Reg::Neg(positive, score, full);
        Reg::Cast<int32_t, float, toInt>(token, positive, valid);
        Reg::Scatter<int32_t, uint32_t>(uniqueUb, token, uid, head);
        Reg::Duplicate(payload, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(payload, recU, odd, valid);

        Reg::ShiftLefts(packed, uid, int16_t{16}, full);
        Reg::Add(packed, packed, payload, full);
        Reg::Store(recU + base, packed, count);
        Reg::Duplicate(lastIndex, uint32_t{63});
        Reg::Gather<uint32_t, uint32_t>(carry, prefix, lastIndex);
        Reg::Duplicate(lastIndex, count - 1U);
        Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
    }

    Reg::Compares<uint32_t, CMPMODE::EQ>(head, lane, uint32_t{32}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(ctxUb), carry, lane, head);
}

// P = 1, build: a single sorted row already holds distinct tokens, so the row
// is the unique list. Initialises owner to -1 and zeroes the flags.
__simt_callee__ __aicore__ inline void WsBuildDirectRow(
    uint32_t row, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    __gm__ int32_t* posGm, __ubuf__ int32_t* uniqueUb, __ubuf__ int32_t* ownerUb,
    __ubuf__ int32_t* flagsUb, __ubuf__ int32_t* scanUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t rowOff = row * topkDim;

    int32_t guardSum = 0;
    for (uint32_t j = t; j < topkDim; j += ECHO_WS_VF_THREADS) {
        if (j < effTopk) {
            int32_t tok = posGm[rowOff + j];
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            uniqueUb[j] = tokSafe;
            if (j > 0u) {
                int32_t prev = posGm[rowOff + j - 1u];
                if (tok <= prev) guardSum += 1;
            }
        }
        ownerUb[j] = -1;
    }
    if (guardSum > 0) scanUb[65] = guardSum;
}

// Zeroes the flags segment of the workset ([0, B]).
__simt_callee__ __aicore__ inline void WsClearFlags(
    uint32_t B, __ubuf__ int32_t* flagsUb) {
    for (uint32_t s = threadIdx.x; s <= B; s += ECHO_WS_VF_THREADS) {
        flagsUb[s] = 0;
    }
}

// P > 1, joint initialisation before the scan: owner = -1 and flags = 0.
__simt_callee__ __aicore__ inline void WsJointInitRow(
    uint32_t topkDim, uint32_t B, uint32_t P,
    __ubuf__ int32_t* ownerUb, __ubuf__ int32_t* flagsUb, uint32_t clearFlags) {
    uint32_t n = P * topkDim;
    for (uint32_t i = threadIdx.x; i < n; i += ECHO_WS_VF_THREADS) {
        ownerUb[i] = -1;
    }

    if (clearFlags != 0u) WsClearFlags(B, flagsUb);
}

// Population count of a 32-bit word.
__simt_callee__ __aicore__ inline uint32_t Popcnt32W(uint32_t x) {
    x = x - ((x >> 1u) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
    x = (x + (x >> 4u)) & 0x0F0F0F0Fu;
    x = (x + (x >> 8u)) & 0x00FF00FFu;
    return (x + (x >> 16u)) & 0x000000FFu;
}

// Residency-driven scan (P > 1): walks the lru order of the resident slots,
// finds each slot token in the unique list (prefix bucket table plus a short
// binary search) and records the owning slot for the hit queries. The
// candidate and hit masks are produced with warp ballots.
__simt_callee__ __aicore__ inline void WsScanRow(
    uint32_t row, uint32_t B, uint32_t occ, uint32_t u, uint32_t maxModelLenP1,
    __gm__ int32_t* lruGm, __gm__ int32_t* dthGm,
    __ubuf__ int32_t* uniqueUb, __ubuf__ int32_t* ownerUb,
    __ubuf__ int32_t* flagsUb, __ubuf__ int32_t* scanUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t lruRowOff = row * ECHO_LRU_ROW_PITCH(B);
    uint32_t dthRowOff = row * (B + 1u);

    uint32_t shift = 0u;
    uint32_t lastTok = (maxModelLenP1 > 0u) ? (maxModelLenP1 - 1u) : 0u;
    while ((lastTok >> shift) >= 256u) ++shift;
    __ubuf__ int32_t* bktUb = scanUb + ECHO_WS_RB_OFF;
    for (uint32_t b = t; b <= 256u; b += ECHO_WS_VF_THREADS) {
        uint32_t lo = 0u, hi = u;
        if (b == 256u) {
            lo = u; hi = u;
        } else {
            uint32_t key = b << shift;
            while (lo < hi) {
                uint32_t mid = (lo + hi) >> 1;
                if (uniqueUb[mid] < (int32_t)key) lo = mid + 1u;
                else hi = mid;
            }
        }
        bktUb[b] = static_cast<int32_t>(lo);
    }

    if (occ != 0u) {
        uint32_t warmStride = (B + ECHO_WS_VF_THREADS - 1u) / ECHO_WS_VF_THREADS;
        uint32_t wLo = t * warmStride;
        uint32_t wHi = wLo + warmStride;
        if (wHi > B) wHi = B;
        int32_t preCnt = 0;
        for (uint32_t k = wLo; k < wHi; ++k) {
            preCnt += (dthGm[dthRowOff + 1u + k] > 0) ? 1 : 0;
        }
        if (preCnt < 0) scanUb[65] = preCnt;
    }
    asc_syncthreads();

    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5u;
    const uint32_t warpCount = ECHO_WS_VF_THREADS >> 5u;
    const uint32_t nWords = (B + 31u) >> 5u;
    const uint32_t wpw = (nWords + warpCount - 1u) / warpCount;
    uint32_t swLo = warp * wpw;
    uint32_t swHi = (swLo >= nWords || swLo + wpw > nWords) ? nWords : (swLo + wpw);
    const bool fuseMask = (B <= 8192u);
    __ubuf__ int32_t* fMaskUb = scanUb + ECHO_WS_RB_OFF + ECHO_WS_FMASK_OFF;
    uint32_t myCand = 0u, myHit = 0u;
    if (fuseMask) {
        for (uint32_t w = swLo; w < swHi; ++w) {
            uint32_t order = (w << 5u) + lane;
            bool isCand = false, isHit = false;
            if (order < occ) {
                int32_t slot = lruGm[lruRowOff + order];
                int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
                int32_t tok = dthGm[dthRowOff + (uint32_t)sv];
                if (tok == ECHO_WS_I32_MAX) {
                    flagsUb[(uint32_t)sv] = 2;
                } else {

                    uint32_t bkt = ((uint32_t)tok) >> shift;
                    if (bkt > 255u) bkt = 255u;
                    uint32_t lo = (uint32_t)bktUb[bkt];
                    uint32_t hi = (uint32_t)bktUb[bkt + 1u];
                    if (hi > u) hi = u;
                    if (lo > hi) { lo = 0u; hi = u; }
                    while (lo < hi) {
                        uint32_t mid = (lo + hi) >> 1;
                        if (uniqueUb[mid] < tok) lo = mid + 1u;
                        else hi = mid;
                    }
                    if (lo < u && uniqueUb[lo] == tok) {

                        asc_atomic_max(ownerUb + lo,
                                       (int32_t)((order << 16) | (uint32_t)sv));
                        flagsUb[(uint32_t)sv] = 1;
                        isHit = true;
                    } else {
                        isCand = true;
                    }
                }
            }
            uint32_t cm = asc_ballot(isCand ? 1 : 0);
            uint32_t hm = asc_ballot(isHit ? 1 : 0);
            if (lane == 0u) {
                fMaskUb[w] = static_cast<int32_t>(cm);
                fMaskUb[nWords + w] = static_cast<int32_t>(hm);
                myCand += Popcnt32W(cm);
                myHit += Popcnt32W(hm);
            }
        }
        if (lane == 0u) {
            scanUb[warp] = static_cast<int32_t>(myCand | (myHit << 16));
        }
    } else {

        for (uint32_t order = t; order < occ; order += ECHO_WS_VF_THREADS) {
            int32_t slot = lruGm[lruRowOff + order];
            int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
            int32_t tok = dthGm[dthRowOff + (uint32_t)sv];
            if (tok == ECHO_WS_I32_MAX) {
                flagsUb[(uint32_t)sv] = 2;
                continue;
            }
            uint32_t bkt = ((uint32_t)tok) >> shift;
            if (bkt > 255u) bkt = 255u;
            uint32_t lo = (uint32_t)bktUb[bkt];
            uint32_t hi = (uint32_t)bktUb[bkt + 1u];
            if (hi > u) hi = u;
            if (lo > hi) { lo = 0u; hi = u; }
            while (lo < hi) {
                uint32_t mid = (lo + hi) >> 1;
                if (uniqueUb[mid] < tok) lo = mid + 1u;
                else hi = mid;
            }
            if (lo < u && uniqueUb[lo] == tok) {
                asc_atomic_max(ownerUb + lo,
                               (int32_t)((order << 16) | (uint32_t)sv));
                flagsUb[(uint32_t)sv] = 1;
            }
        }
    }
}

// Residency-driven scan (P = 1): same decision as WsScanRow, with dth read
// from a UB window that the shell preloaded with MTE.
__simt_callee__ __aicore__ inline void WsScanRowUb(
    uint32_t row, uint32_t B, uint32_t occ, uint32_t u, uint32_t maxModelLenP1,
    __gm__ int32_t* lruSrc, __ubuf__ int32_t* dthUb,
    __ubuf__ int32_t* uniqueUb, __ubuf__ int32_t* ownerUb,
    __ubuf__ int32_t* flagsUb, __ubuf__ int32_t* scanUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t lruRowOff = row * ECHO_LRU_ROW_PITCH(B);
    uint32_t dthRowOff = row * (B + 1u);

    uint32_t lv = 0u;
    while ((1u << lv) < (u + 1u)) ++lv;
    uint32_t iters = (occ + ECHO_WS_VF_THREADS - 1u) / ECHO_WS_VF_THREADS;
    constexpr uint32_t W = ECHO_WS_SCAN_BATCH;
    for (uint32_t i0 = 0u; i0 < iters; i0 += W) {
        uint32_t lo[W], hi[W];
        int32_t sv[W], tok[W];

        for (uint32_t k = 0u; k < W; ++k) {
            uint32_t o = t + (i0 + k) * ECHO_WS_VF_THREADS;
            bool act = o < occ;
            int32_t sl = 0;
            if (act) {
                sl = lruSrc[lruRowOff + o];

            }
            sv[k] = sl < 0 ? 0 : (sl > (int32_t)B ? (int32_t)B : sl);

            tok[k] = act ? dthUb[dthRowOff + (uint32_t)sv[k]] : 0;
            lo[k] = 0u;
            hi[k] = (act && tok[k] != ECHO_WS_I32_MAX) ? u : 0u;
        }

        for (uint32_t step = 0u; step < lv; ++step) {
            int32_t pv[W];
            uint32_t mid[W];
            for (uint32_t k = 0u; k < W; ++k) {
                mid[k] = (lo[k] + hi[k]) >> 1;
                pv[k] = uniqueUb[mid[k]];
            }
            for (uint32_t k = 0u; k < W; ++k) {
                if (lo[k] < hi[k]) {
                    if (pv[k] < tok[k]) lo[k] = mid[k] + 1u;
                    else hi[k] = mid[k];
                }
            }
        }

        for (uint32_t k = 0u; k < W; ++k) {
            uint32_t o = t + (i0 + k) * ECHO_WS_VF_THREADS;
            if (o >= occ) continue;
            if (tok[k] == ECHO_WS_I32_MAX) {
                flagsUb[(uint32_t)sv[k]] = 2;
                continue;
            }
            if (lo[k] < u && uniqueUb[lo[k]] == tok[k]) {

                asc_atomic_max(ownerUb + lo[k], (int32_t)((o << 16) | (uint32_t)sv[k]));
                flagsUb[(uint32_t)sv[k]] = 1;
            }

        }
    }
}

// P = 1, finalize: maps every query to its slot (current_slots), counts the
// misses and writes the ordered miss list.
__simt_callee__ __aicore__ inline int32_t WsFinalizeFlatRow(
    uint32_t row, uint32_t topkDim, uint32_t effTopk, uint32_t B,
    int32_t avail, __gm__ int32_t* curSlotsGm, __gm__ int32_t* wsMissRankGm,
    __gm__ int32_t* numMissesGm, __gm__ int32_t* freeSizeGm,
    __ubuf__ int32_t* ownerUb, __ubuf__ int32_t* uniqueUb, __ubuf__ int32_t* scanUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t rowOff = row * topkDim;
    uint32_t W = (topkDim + ECHO_WS_VF_THREADS - 1u) / ECHO_WS_VF_THREADS;
    uint32_t jLo = t * W;
    uint32_t jHi = (jLo + W < topkDim) ? (jLo + W) : topkDim;

    int32_t cnt = 0;
    for (uint32_t j = jLo; j < jHi; ++j) {
        if (j >= effTopk) {
            curSlotsGm[rowOff + j] = 0;
            continue;
        }
        int32_t tok = uniqueUb[j];
        if (tok == 0) {
            curSlotsGm[rowOff + j] = 0;
        } else {
            int32_t ow = ownerUb[j];
            if (ow >= 0) curSlotsGm[rowOff + j] = ow & 0xFFFF;
            else { curSlotsGm[rowOff + j] = ECHO_WS_I32_MAX; cnt += 1; }
        }
    }

    int32_t excl = WsExclusiveScan(cnt, scanUb);
    uint32_t run = 0u;
    for (uint32_t j = jLo; j < jHi; ++j) {
        if (j >= effTopk) continue;
        int32_t tok = uniqueUb[j];
        if (tok == 0) continue;
        if (ownerUb[j] < 0) {
            wsMissRankGm[rowOff + (uint32_t)(excl + (int32_t)run)] = (int32_t)j;
            run += 1u;
        }
    }

    uint32_t warpCount = ECHO_WS_VF_THREADS >> 5;
    int32_t nm = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) nm += scanUb[w];
    if (nm < 0) nm = 0;
    if (t == 0u) {
        numMissesGm[row] = nm;
        int32_t fs = nm - avail;
        if (fs < 0) fs = 0;
        freeSizeGm[row] = fs;
        if (fs != 0) scanUb[64] = 1;
    }
    return nm;
}

// P > 1, finalize (SIMT form): resolves each flat id to its slot and writes
// current_slots and the miss list in token order.
__simt_callee__ __aicore__ inline int32_t WsFinalizeJointRow(
    uint32_t row, uint32_t topkDim, uint32_t effTopk, uint32_t P, uint32_t B,
    int32_t avail, __gm__ int32_t* curSlotsGm, __gm__ int32_t* freeIdxGm,
    __gm__ int32_t* wsMissRankGm, __gm__ int32_t* numMissesGm, __gm__ int32_t* freeSizeGm,
    __ubuf__ int32_t* mapUb, __ubuf__ int32_t* ownerUb, __ubuf__ int32_t* uniqueUb,
    __ubuf__ int32_t* scanUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t n = P * effTopk;
    uint32_t flatDim = P * topkDim;
    uint32_t rowOff = row * flatDim;
    __gm__ int32_t* stashGm = freeIdxGm + rowOff;
    uint32_t W = (n + ECHO_WS_VF_THREADS - 1u) / ECHO_WS_VF_THREADS;
    uint32_t iLo = t * W;
    uint32_t iHi = (iLo + W < n) ? (iLo + W) : n;

    int32_t cnt = 0;
    for (uint32_t i = iLo; i < iHi; ++i) {
        uint32_t flat = (uint32_t)stashGm[i];
        int32_t uid = mapUb[flat];
        int32_t tok = uniqueUb[(uint32_t)uid];
        if (tok == 0) {
            curSlotsGm[rowOff + flat] = 0;
        } else {
            int32_t ow = ownerUb[(uint32_t)uid];
            if (ow >= 0) curSlotsGm[rowOff + flat] = ow & 0xFFFF;
            else { curSlotsGm[rowOff + flat] = ECHO_WS_I32_MAX; cnt += 1; }
        }
    }

    int32_t excl = WsExclusiveScan(cnt, scanUb);
    uint32_t run = 0u;
    for (uint32_t i = iLo; i < iHi; ++i) {
        uint32_t flat = (uint32_t)stashGm[i];
        int32_t uid = mapUb[flat];
        if (uniqueUb[(uint32_t)uid] == 0) continue;
        if (ownerUb[(uint32_t)uid] < 0) {
            wsMissRankGm[rowOff + (uint32_t)(excl + (int32_t)run)] = (int32_t)flat;
            run += 1u;
        }
    }

    uint32_t warpCount = ECHO_WS_VF_THREADS >> 5;
    int32_t nm = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) nm += scanUb[w];
    if (nm < 0) nm = 0;
    if (t == 0u) {
        numMissesGm[row] = nm;
        int32_t fs = nm - avail;
        if (fs < 0) fs = 0;
        freeSizeGm[row] = fs;
        if (fs != 0) scanUb[64] = 1;
    }
    return nm;
}

// P > 1, finalize (SIMD form): vector version of WsFinalizeJointRow.
__simd_vf__ inline void WsFinalizeVf(
    __ubuf__ int32_t* recUb, __ubuf__ int32_t* uniqueUb, __ubuf__ int32_t* ownerUb,
    __ubuf__ int32_t* ctxUb, uint32_t n, uint32_t topkDim, uint32_t effTopk, uint32_t P,
    uint32_t flatMask) {
    using namespace AscendC;
    __ubuf__ uint32_t* recU = reinterpret_cast<__ubuf__ uint32_t*>(recUb);
    Reg::RegTensor<uint32_t> lane, zero, one, max16, i32max;
    Reg::RegTensor<uint32_t> idx, packed, flat, flatL, flatMaskT, uid, slotU, cs16, csHi, res, resVal;
    Reg::RegTensor<uint32_t> vflag, rank, inc, mc, tmp63;
    Reg::RegTensor<int32_t> tok, ow, zeroI;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg valid, tokZero, owGe0, missM, missN, chosen;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(max16, uint32_t{0xFFFF});

    Reg::Duplicate(flatMaskT, flatMask);
    Reg::Duplicate(i32max, uint32_t{0x7FFFFFFF});
    Reg::Duplicate(zeroI, int32_t{0});
    Reg::Duplicate(mc, uint32_t{0});

    for (uint32_t base = 0; base < n; base += 64U) {
        uint32_t remaining = n - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);
        const uint32_t count = n - base < 64U ? n - base : 64U;
        Reg::Adds(idx, lane, base, full);
        Reg::Duplicate(packed, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(packed, recU, idx, valid);
        Reg::And(flat, packed, max16, full);
        Reg::ShiftRights(uid, packed, int16_t{16}, full);
        Reg::Duplicate(tok, int32_t{0});
        Reg::Gather<int32_t, int32_t, uint32_t>(tok, uniqueUb, uid, valid);
        Reg::Duplicate(ow, int32_t{0});
        Reg::Gather<int32_t, int32_t, uint32_t>(ow, ownerUb, uid, valid);
        Reg::Compares<int32_t, CMPMODE::EQ>(tokZero, tok, int32_t{0}, valid);
        Reg::Compares<int32_t, CMPMODE::GE>(owGe0, ow, int32_t{0}, valid);
        Reg::And(slotU, reinterpret_cast<Reg::RegTensor<uint32_t>&>(ow), max16, full);
        Reg::Select(cs16, slotU, max16, owGe0);
        Reg::Select(cs16, zero, cs16, tokZero);
        Reg::ShiftLefts(csHi, cs16, int16_t{16}, full);
        Reg::Add(res, csHi, flat, full);
        Reg::Store(recU + base, res, count);
    }

    Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();

    for (uint32_t base = 0; base < n; base += 64U) {
        uint32_t remaining = n - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);
        Reg::Adds(idx, lane, base, full);
        Reg::Duplicate(res, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(res, recU, idx, valid);
        Reg::And(flat, res, max16, full);
        Reg::ShiftRights(cs16, res, int16_t{16}, full);
        Reg::Compares<uint32_t, CMPMODE::EQ>(missM, cs16, uint32_t{0xFFFF}, full);
        Reg::Select(vflag, one, zero, missM);
        Reg::Select(vflag, vflag, zero, valid);
        Reg::Compares<uint32_t, CMPMODE::NE>(missN, vflag, uint32_t{0}, full);
        Reg::Unsqueeze<uint32_t>(rank, missN);
        Reg::Add(rank, rank, mc, full);

        Reg::Select(resVal, i32max, cs16, missM);
        Reg::And(flatL, flat, flatMaskT, full);
        Reg::Scatter<uint32_t, uint32_t>(
            reinterpret_cast<__ubuf__ uint32_t*>(ownerUb), resVal, flatL, valid);

        Reg::Scatter<uint32_t, uint32_t>(
            reinterpret_cast<__ubuf__ uint32_t*>(uniqueUb), flat, rank, missN);

        Reg::Add(inc, rank, vflag, full);
        Reg::Duplicate(tmp63, uint32_t{63});
        Reg::Gather<uint32_t, uint32_t>(mc, inc, tmp63);
    }

    if (effTopk < topkDim) {
        for (uint32_t p = 0u; p < P; ++p) {
            for (uint32_t j = effTopk; j < topkDim; j += 64U) {
                uint32_t padCnt = (topkDim - j < 64U) ? (topkDim - j) : 64U;
                Reg::Store(ownerUb + p * topkDim + j, zeroI, padCnt);
            }
        }
    }

    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{33}, full);
    Reg::Scatter<uint32_t, uint32_t>(
        reinterpret_cast<__ubuf__ uint32_t*>(ctxUb), mc, lane, chosen);
}

// Returns mask with only its k lowest set bits kept.
__simt_callee__ __aicore__ inline uint32_t FirstKSetBitsW(uint32_t mask, uint32_t k) {
    uint32_t surv = mask;
    for (uint32_t r = 0u; r < k; ++r) surv &= surv - 1u;
    return mask & ~surv;
}

// Emits the victim / survivor / hit masks and counts consumed by kernel 2
// from the flags and the lru order.
__simt_callee__ __aicore__ inline void OpClassifyMaskEmit2(
    uint32_t B, uint32_t fsClamp, uint32_t occ,
    __gm__ int32_t* masksOut, __gm__ int32_t* countsOut, __gm__ int32_t* rbBaseOut,
    __ubuf__ int32_t* flagsUb, __gm__ int32_t* lruSrc, __ubuf__ int32_t* scanScratch,
    uint32_t tailFence) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t warpCount = ECHO_WS_VF_THREADS >> 5;
    uint32_t nWords = (B + 31u) / 32u;

    if (occ == 0u && fsClamp == 0u) {
        for (uint32_t i = t; i < 3u * nWords; i += ECHO_WS_VF_THREADS) {
            masksOut[i] = 0;
        }
        if (t == 0u) {
            rbBaseOut[0] = 0;
            for (uint32_t c = 0; c < 8u; ++c) countsOut[c] = 0;
        }
        return;
    }

    uint32_t lane = t & 31u;
    uint32_t warp = t >> 5;
    uint32_t wpw = (nWords + warpCount - 1u) / warpCount;
    uint32_t wLo = warp * wpw;
    uint32_t wHi = (wLo >= nWords || wLo + wpw > nWords) ? nWords : (wLo + wpw);
    uint32_t myWords = (wHi > wLo) ? (wHi - wLo) : 0u;
    uint32_t candPk = 0u, hitPk = 0u, laneCand = 0u, laneHit = 0u;
    for (uint32_t wi = 0u; wi < myWords; ++wi) {
        uint32_t p = ((wLo + wi) << 5) + lane;
        bool isCand = false, isHit = false;
        if (p < occ) {
            int32_t slot = lruSrc[p];
            int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
            int32_t v = flagsUb[sv];
            isHit = (v & 1) != 0;
            isCand = !isHit && ((v & 2) == 0);
        }
        if (wpw <= 32u) {
            candPk |= isCand ? (1u << wi) : 0u;
            hitPk |= isHit ? (1u << wi) : 0u;
        } else {
            laneCand += isCand ? 1u : 0u;
            laneHit += isHit ? 1u : 0u;
        }
    }

    uint32_t myCand = (wpw <= 32u) ? Popcnt32W(candPk) : laneCand;
    uint32_t myHit = (wpw <= 32u) ? Popcnt32W(hitPk) : laneHit;
    int32_t warpCand = asc_reduce_add((int32_t)myCand);
    int32_t warpHit = asc_reduce_add((int32_t)myHit);
    if (lane == 0u) scanScratch[warp] = warpCand | (warpHit << 16);
    asc_syncthreads();

    uint32_t baseC = 0u;
    for (uint32_t w = 0u; w < warp; ++w) baseC += (uint32_t)scanScratch[w] & 0xFFFFu;
    int32_t grand = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) grand += scanScratch[w];
    uint32_t totalCand = (uint32_t)grand & 0xFFFFu;
    uint32_t totalH = (uint32_t)grand >> 16;
    uint32_t victims = (fsClamp < totalCand) ? fsClamp : totalCand;
    uint32_t totalK = totalCand - victims;
    uint32_t rbBase = totalK + totalH;

    uint32_t C = baseC;
    for (uint32_t wi = 0u; wi < myWords; ++wi) {
        uint32_t w = wLo + wi;
        bool isCand = false, isHit = false;
        if (wpw <= 32u) {
            isCand = ((candPk >> wi) & 1u) != 0u;
            isHit = ((hitPk >> wi) & 1u) != 0u;
        } else {
            uint32_t p = (w << 5) + lane;
            if (p < occ) {
                int32_t slot = lruSrc[p];
                int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
                int32_t v = flagsUb[sv];
                isHit = (v & 1) != 0;
                isCand = !isHit && ((v & 2) == 0);
            }
        }
        uint32_t candMask = asc_ballot(isCand ? 1 : 0);
        uint32_t hitMask = asc_ballot(isHit ? 1 : 0);
        uint32_t cnt = Popcnt32W(candMask);
        uint32_t k = (fsClamp > C) ? ((fsClamp - C < cnt) ? (fsClamp - C) : cnt) : 0u;
        uint32_t victimMask = (k >= cnt) ? candMask : FirstKSetBitsW(candMask, k);
        uint32_t survMask = candMask & ~victimMask;
        if (lane == 0u) {
            masksOut[w] = (int32_t)victimMask;
            masksOut[nWords + w] = (int32_t)survMask;
            masksOut[2u * nWords + w] = (int32_t)hitMask;
        }
        C += cnt;
    }

    if (t == 0u) {
        rbBaseOut[0] = (int32_t)rbBase;
        countsOut[0] = (int32_t)totalCand;
        countsOut[1] = (int32_t)totalH;
        countsOut[2] = (int32_t)victims;
        countsOut[3] = (int32_t)rbBase;
        countsOut[4] = (int32_t)totalK;
        countsOut[5] = (int32_t)fsClamp;
        countsOut[6] = 0;
        countsOut[7] = 0;
    }

    if (tailFence != 0u) asc_syncthreads();
}

// Mask emission for rows whose scan already produced candidate and hit masks:
// prefix sums, the victim / survivor split and the three mask writes.
__simt_callee__ __aicore__ inline void OpEmitMasksFused(
    uint32_t B, uint32_t fsClamp, uint32_t occ,
    __gm__ int32_t* masksOut, __gm__ int32_t* countsOut, __gm__ int32_t* rbBaseOut,
    __ubuf__ int32_t* scanScratch, __ubuf__ int32_t* fMaskUb) {
    uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t warpCount = ECHO_WS_VF_THREADS >> 5;
    uint32_t nWords = (B + 31u) / 32u;
    if (occ == 0u && fsClamp == 0u) {
        for (uint32_t i = t; i < 3u * nWords; i += ECHO_WS_VF_THREADS) {
            masksOut[i] = 0;
        }
        if (t == 0u) {
            rbBaseOut[0] = 0;
            for (uint32_t c = 0; c < 8u; ++c) countsOut[c] = 0;
        }
        return;
    }
    uint32_t lane = t & 31u;
    uint32_t warp = t >> 5;
    uint32_t wpw = (nWords + warpCount - 1u) / warpCount;
    uint32_t wLo = warp * wpw;
    uint32_t wHi = (wLo >= nWords || wLo + wpw > nWords) ? nWords : (wLo + wpw);

    uint32_t baseC = 0u;
    for (uint32_t w = 0u; w < warp; ++w) baseC += (uint32_t)scanScratch[w] & 0xFFFFu;
    int32_t grand = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) grand += scanScratch[w];
    uint32_t totalCand = (uint32_t)grand & 0xFFFFu;
    uint32_t totalH = (uint32_t)grand >> 16;
    uint32_t victims = (fsClamp < totalCand) ? fsClamp : totalCand;
    uint32_t totalK = totalCand - victims;
    uint32_t rbBase = totalK + totalH;

    uint32_t C = baseC;
    for (uint32_t w = wLo; w < wHi; ++w) {
        uint32_t cand = (uint32_t)fMaskUb[w];
        uint32_t hit = (uint32_t)fMaskUb[nWords + w];
        uint32_t cnt = Popcnt32W(cand);
        uint32_t k = (fsClamp > C) ? ((fsClamp - C < cnt) ? (fsClamp - C) : cnt) : 0u;
        uint32_t victimMask = (k >= cnt) ? cand : FirstKSetBitsW(cand, k);
        uint32_t survMask = cand & ~victimMask;
        if (lane == 0u) {
            masksOut[w] = (int32_t)victimMask;
            masksOut[nWords + w] = (int32_t)survMask;
            masksOut[2u * nWords + w] = (int32_t)hit;
        }
        C += cnt;
    }
    if (t == 0u) {
        rbBaseOut[0] = (int32_t)rbBase;
        countsOut[0] = (int32_t)totalCand;
        countsOut[1] = (int32_t)totalH;
        countsOut[2] = (int32_t)victims;
        countsOut[3] = (int32_t)rbBase;
        countsOut[4] = (int32_t)totalK;
        countsOut[5] = (int32_t)fsClamp;
        countsOut[6] = 0;
        countsOut[7] = 0;
    }
}

__simt_callee__ __aicore__ inline void WsFrontPhases(
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t maxModelLenP1,
    uint32_t freeStackSize, uint32_t numRequests, uint32_t specOn,
    __gm__ int32_t* htdGm, __gm__ int32_t* dthGm, __gm__ int32_t* lruGm,
    __gm__ int32_t* freeGm, __gm__ int32_t* availGm, __gm__ int32_t* fifoGm,
    __gm__ int32_t* splGm, __gm__ uint8_t* resetMaskGm, __gm__ uint8_t* freshGm,
    __gm__ int32_t* wsGm, __ubuf__ int32_t* scanUb) {

    __gm__ int32_t* wsCumsumGm = wsGm;
    __gm__ int32_t* pushCountGm = wsGm + numRequests * (B + 1u);

    if (specOn != 0u) OpFillZero2(rowStart, rowEnd, pushCountGm);

    bool anyReset = false;
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        anyReset = anyReset || (resetMaskGm[row] != 0);
    }
    if (anyReset) {
        OpResetRows2(rowStart, rowEnd, maxModelLenP1, resetMaskGm, htdGm);
        OpResetDthLru2(rowStart, rowEnd, B, resetMaskGm, dthGm, lruGm);
        OpResetFree2(rowStart, rowEnd, B, freeStackSize, resetMaskGm, freeGm);
        OpResetAvailFifo2(rowStart, rowEnd, B, resetMaskGm, availGm, fifoGm);
        WsSyncGrid(wsGm);
    }

    bool anyFresh = false;
    bool allFresh = true;
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        uint8_t fv = freshGm[row];
        anyFresh = anyFresh || (fv != 0u);
        allFresh = allFresh && (fv != 0u);
    }
    if (threadIdx.x == 0u) scanUb[66] = allFresh ? 1 : 0;
    if (anyFresh) OpFreshClear2(rowStart, rowEnd, freshGm);

    if (specOn != 0u) {
        OpInvalidateSpecSuffix2(rowStart, rowEnd, B, maxModelLenP1, splGm, dthGm, htdGm, wsCumsumGm);
        WsSyncGrid(wsGm);
        OpSpecCumsum2(rowStart, rowEnd, B, wsCumsumGm, pushCountGm);
        WsSyncGrid(wsGm);
        OpSpecPushFree2(rowStart, rowEnd, B, freeStackSize, wsCumsumGm, availGm, freeGm);
        WsSyncGrid(wsGm);
        OpSpecAvailAdd2(rowStart, rowEnd, availGm, pushCountGm);
        WsSyncGrid(wsGm);
    }
    asc_syncthreads();
}

// Main frame for P = 1: scans all rows of the block in a loop inside one
// frame call.
__simt_vf__ __aicore__ LAUNCH_BOUND(ECHO_WS_VF_THREADS) inline void EchoWsVfFrame(
    __gm__ int32_t* htdGm, __gm__ int32_t* dthGm, __gm__ int32_t* lruGm,
    __gm__ int32_t* freeGm, __gm__ int32_t* availGm, __gm__ int32_t* fifoGm,
    __gm__ int32_t* posGm,
    __gm__ uint8_t* freshGm,
    __gm__ int32_t* curSlotsGm, __gm__ int32_t* missCountGm,
    __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* wsGm, __ubuf__ int32_t* ubBase, __ubuf__ int32_t* dthUb,
    uint32_t chunkStart, uint32_t chunkEnd, uint32_t firstChunk, uint32_t blockId,
    uint32_t B, uint32_t topkDim, uint32_t maxModelLenP1, uint32_t freeStackSize,
    uint32_t numRequests, uint32_t specOn) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    const uint32_t P = 1u;
    const uint32_t effTopk = (topkDim < B) ? topkDim : B;
    const uint32_t numTokens = numRequests * P;

    __gm__ int32_t* ws = wsGm;
    uint32_t o = 0u;
    o += numRequests * (B + 1u);
    o += numRequests;
    __gm__ int32_t* numHitsGm = ws + o; o += numRequests;
    __gm__ int32_t* numMissesGm = ws + o; o += numRequests;
    o += numRequests;
    o += numTokens * topkDim;
    __gm__ int32_t* freeSizeGm = ws + o; o += numRequests;
    o += numTokens * topkDim;
    __gm__ int32_t* wsMissRankGm = ws + o; o += numTokens * topkDim;
    o = (o + 7u) & ~7u;
    uint32_t maskRowElems = 3u * ((B + 31u) / 32u);
    __gm__ int32_t* masksGm = ws + o; o += numRequests * maskRowElems;
    __gm__ int32_t* countsGm = ws + o; o += numRequests * 8u;
    __gm__ int32_t* handoffGm = ws + o;

    __ubuf__ int32_t* scanUb = ubBase;
    __ubuf__ int32_t* rbUb = ubBase + ECHO_WS_RB_OFF;
    __ubuf__ int32_t* uniqueUb = ubBase + ECHO_WS_UNIQUE_OFF(topkDim, P);
    __ubuf__ int32_t* ownerUb = ubBase + ECHO_WS_OWNER_OFF(topkDim, P);

    __ubuf__ int32_t* flagsUb = ubBase + ECHO_WS_B_OFF(topkDim, P) + ECHO_WS_ALIGN8(P * topkDim);

    if (t == 0u && firstChunk != 0u) scanUb[64] = 0;
    asc_syncthreads();

    if (scanUb[66] != 0 && specOn == 0u) {
        for (uint32_t row = chunkStart; row < chunkEnd; ++row) {
            FreshLoadRow(row, B, topkDim, effTopk, maxModelLenP1, freeStackSize,
                         availGm, fifoGm, freshGm, freeGm, posGm, curSlotsGm,
                         missCountGm, missTokensGm, missSlotsGm, htdGm, dthGm, lruGm);
        }
        OpColdScratchZero2(chunkStart, chunkEnd, maskRowElems, masksGm, countsGm);
        if (t == 0u) {
            handoffGm[blockId * 2u + 1u] = 1;
            handoffGm[blockId * 2u + 0u] = 0;
        }
        return;
    }

    __gm__ int32_t* lruSrc = lruGm + chunkStart * ECHO_LRU_ROW_PITCH(B);
    const uint32_t chunkRows = chunkEnd - chunkStart;
    for (uint32_t r = 0u; r < chunkRows; ++r) {
        uint32_t row = chunkStart + r;
        int32_t avail = availGm[row];
        uint32_t occ = (avail >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)avail);
#ifndef ECHO_WS_ABLATE_PACK

        WsBuildDirectRow(row, topkDim, effTopk, maxModelLenP1, posGm, uniqueUb, ownerUb, flagsUb, scanUb);
#endif
        WsClearFlags(B, flagsUb);
        asc_syncthreads();
#ifndef ECHO_WS_ABLATE_SCAN

        WsScanRowUb(r, B, occ, effTopk, maxModelLenP1, lruSrc, dthUb,
                    uniqueUb, ownerUb, flagsUb, scanUb);
#endif
        asc_syncthreads();
#ifndef ECHO_WS_ABLATE_FINALIZE

        int32_t nm = WsFinalizeFlatRow(row, topkDim, effTopk, B, avail, curSlotsGm, wsMissRankGm,
                                       numMissesGm, freeSizeGm, ownerUb, uniqueUb, scanUb);
#else
        int32_t nm = 0;
#endif

        int32_t fs = nm - avail;
        if (fs < 0) fs = 0;
        uint32_t fsClamp = (fs <= 0) ? 0u : ((fs > (int32_t)topkDim) ? topkDim : (uint32_t)fs);
#ifndef ECHO_WS_ABLATE_MASK

        OpClassifyMaskEmit2(B, fsClamp, occ,
                            masksGm + row * maskRowElems, countsGm + row * 8u,
                            numHitsGm + row, flagsUb,
                            lruGm + row * ECHO_LRU_ROW_PITCH(B), rbUb, 1u);
#endif
    }

    if (t == 0u) {
        handoffGm[blockId * 2u + 0u] = (scanUb[64] != 0) ? 1 : 0;
        handoffGm[blockId * 2u + 1u] = 0;
    }
}

// P > 1, pack frame: packs the first row into the merge area (the following
// rows are packed at the tail of the preceding joint frame).
__simt_vf__ __aicore__ LAUNCH_BOUND(ECHO_WS_VF_THREADS) inline void EchoWsPackFrame(
    __gm__ int32_t* posGm, __gm__ int32_t* curSlotsGm, __ubuf__ int32_t* ubBase,
    uint32_t row, uint32_t topkDim, uint32_t effTopk, uint32_t P,
    uint32_t flatBase, uint32_t posRowStride, uint32_t maxModelLenP1) {
#ifndef ECHO_WS_ABLATE_PACK

    WsPackRow(row, topkDim, effTopk, P, flatBase, posRowStride, maxModelLenP1,
              posGm, curSlotsGm, ubBase + ECHO_WS_A_OFF);
#else
    (void)posGm; (void)curSlotsGm; (void)ubBase; (void)row; (void)topkDim;
    (void)effTopk; (void)P; (void)flatBase; (void)posRowStride; (void)maxModelLenP1;
#endif
}

// P > 1, joint frame: one call per row and phase (initialisation, scan,
// finalize), selected by the phase argument.
__simt_vf__ __aicore__ LAUNCH_BOUND(ECHO_WS_VF_THREADS) inline void EchoWsJointFrame(
    __gm__ int32_t* lruGm, __gm__ int32_t* dthGm, __gm__ int32_t* availGm,
    __gm__ int32_t* curSlotsGm, __gm__ int32_t* posGm, __gm__ int32_t* wsGm,
    __ubuf__ int32_t* ubBase,
    uint32_t row, uint32_t rowEnd, uint32_t blockId, uint32_t B, uint32_t topkDim,
    uint32_t maxModelLenP1, uint32_t numRequests, uint32_t specPositions,
    uint32_t phase,

    uint32_t Pg, uint32_t chunk, uint32_t chunkCount, uint32_t clearFlags,
    uint32_t nextRow, uint32_t nextChunk, uint32_t hasNext,
    __gm__ int32_t* missRun0Gm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    const uint32_t P = specPositions;
    const uint32_t effTopk = (topkDim < B) ? topkDim : B;
    const uint32_t numTokens = numRequests * Pg;

    __gm__ int32_t* ws = wsGm;
    uint32_t o = 0u;
    o += numRequests * (B + 1u);
    o += numRequests;
    __gm__ int32_t* numHitsGm = ws + o; o += numRequests;
    __gm__ int32_t* numMissesGm = ws + o; o += numRequests;
    o += numRequests;
    o += numTokens * topkDim;
    __gm__ int32_t* freeSizeGm = ws + o; o += numRequests;
    o += numTokens * topkDim;
    o += numTokens * topkDim;
    o = (o + 7u) & ~7u;
    uint32_t maskRowElems = 3u * ((B + 31u) / 32u);
    __gm__ int32_t* masksGm = ws + o; o += numRequests * maskRowElems;
    __gm__ int32_t* countsGm = ws + o; o += numRequests * 8u;
    __gm__ int32_t* handoffGm = ws + o;

    __ubuf__ int32_t* scanUb = ubBase;
    __ubuf__ int32_t* rbUb = ubBase + ECHO_WS_RB_OFF;
    __ubuf__ int32_t* uniqueUb = ubBase + ECHO_WS_UNIQUE_OFF(topkDim, P);
    __ubuf__ int32_t* ownerUb = ubBase + ECHO_WS_OWNER_OFF(topkDim, P);

    __ubuf__ int32_t* flagsUb = ubBase + ((chunkCount > 1u)
        ? ECHO_WS_FLAGS_OFF(topkDim, B, P)
        : (ECHO_WS_B_OFF(topkDim, P) + ECHO_WS_ALIGN8(P * topkDim)));

    if (phase == 0u) {

#ifndef ECHO_WS_ABLATE_DEDUP

        uint32_t u = (uint32_t)scanUb[32];
#else
        uint32_t u = 0u;
#endif

        int32_t avail = availGm[row];
        uint32_t occ = (avail >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)avail);
        WsJointInitRow(topkDim, B, P, ownerUb, flagsUb, clearFlags);
        asc_syncthreads();
#ifndef ECHO_WS_ABLATE_SCAN

        WsScanRow(row, B, occ, u, maxModelLenP1, lruGm, dthGm,
                  uniqueUb, ownerUb, flagsUb, scanUb);
#endif

    } else {

#ifndef ECHO_WS_ABLATE_FINALIZE

        int32_t nm = (int32_t)scanUb[33];
#else
        int32_t nm = 0;
#endif

        const uint32_t isLast = (chunk + 1u == chunkCount) ? 1u : 0u;
        int32_t totalNm = nm;
        if (chunkCount > 1u) {
            if (isLast == 0u) {
                if (t == 0u) { scanUb[34] = nm; missRun0Gm[row] = nm; }
            } else {
                totalNm = (int32_t)scanUb[34] + nm;
            }
        }

        int32_t avail = availGm[row];
        uint32_t occ = (avail >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)avail);
        int32_t fs = totalNm - avail;
        if (fs < 0) fs = 0;

        const int32_t fsLimit = (int32_t)(Pg * topkDim);
        uint32_t fsClamp = (fs <= 0) ? 0u : ((fs > fsLimit) ? (uint32_t)fsLimit : (uint32_t)fs);
        if (isLast != 0u) {
            if (t == 0u) scanUb[64] = 0;
#ifndef ECHO_WS_ABLATE_MASK

            if (chunkCount == 1u && B <= 8192u) {

                OpEmitMasksFused(B, fsClamp, occ,
                                 masksGm + row * maskRowElems, countsGm + row * 8u,
                                 numHitsGm + row, scanUb,
                                 scanUb + ECHO_WS_RB_OFF + ECHO_WS_FMASK_OFF);
            } else {
                OpClassifyMaskEmit2(B, fsClamp, occ,
                                    masksGm + row * maskRowElems, countsGm + row * 8u,
                                    numHitsGm + row, flagsUb,
                                    lruGm + row * ECHO_LRU_ROW_PITCH(B), rbUb, 0u);
            }
#endif
            if (t == 0u) {
                numMissesGm[row] = totalNm;
                freeSizeGm[row] = fs;
                if (fs != 0) scanUb[64] = 1;
                handoffGm[blockId * 2u + 0u] = (scanUb[64] != 0) ? 1 : 0;
                handoffGm[blockId * 2u + 1u] = 0;
            }
        }

#ifndef ECHO_WS_ABLATE_PACK

        if (hasNext != 0u) {
            WsPackRow(nextRow, topkDim, effTopk, P,
                      nextChunk * (P * topkDim), Pg * topkDim, maxModelLenP1,
                      posGm, curSlotsGm, ubBase + ECHO_WS_A_OFF);
        }
#endif
    }
}

// Front frame, launched once per block before the row phases: reset, fresh
// handling and speculative-suffix invalidation.
__simt_vf__ __aicore__ LAUNCH_BOUND(ECHO_WS_VF_THREADS) inline void EchoWsFrontFrame(
    __gm__ int32_t* htdGm, __gm__ int32_t* dthGm, __gm__ int32_t* lruGm,
    __gm__ int32_t* freeGm, __gm__ int32_t* availGm, __gm__ int32_t* fifoGm,
    __gm__ int32_t* splGm, __gm__ uint8_t* resetMaskGm, __gm__ uint8_t* freshGm,
    __gm__ int32_t* wsGm, __ubuf__ int32_t* ubBase,
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t maxModelLenP1,
    uint32_t freeStackSize, uint32_t numRequests, uint32_t specOn) {
#ifndef ECHO_WS_ABLATE_FRONT

    WsFrontPhases(rowStart, rowEnd, B, maxModelLenP1, freeStackSize, numRequests, specOn,
                  htdGm, dthGm, lruGm, freeGm, availGm, fifoGm, splGm, resetMaskGm,
                  freshGm, wsGm, ubBase);
#else
    (void)htdGm; (void)dthGm; (void)lruGm; (void)freeGm; (void)availGm; (void)fifoGm;
    (void)splGm; (void)resetMaskGm; (void)freshGm; (void)wsGm; (void)ubBase;
    (void)rowStart; (void)rowEnd; (void)B; (void)maxModelLenP1; (void)freeStackSize;
    (void)numRequests; (void)specOn;
#endif
}

// Event pair: vector work done -> scalar side may read and continue.
__aicore__ inline void WsVectorToScalar(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::V_S>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(id);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::V_S>(id);
}
// Event pair: scalar writes done -> vector side may read.
__aicore__ inline void WsScalarToVector(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::S_V>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(id);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::S_V>(id);
}

// Event pair: vector writes done -> MTE3 may copy them out.
__aicore__ inline void WsVectorToMte3(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(id);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(id);
}
// Event pair: MTE3 copy done -> scalar side may continue.
__aicore__ inline void WsMte3ToScalar(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::MTE3_S>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(id);
}

}

// Kernel entry of kernel 1 (decode step): reads the tiling struct from GM,
// allocates the workset and drives the frames over the rows of the block.
extern "C" __global__ __aicore__ void echo_lru_kernel(
    GM_ADDR htd, GM_ADDR dth, GM_ADDR lru, GM_ADDR free, GM_ADDR avail, GM_ADDR fifo,
    GM_ADDR pos, GM_ADDR spl, GM_ADDR resetMask, GM_ADDR fresh, GM_ADDR curSlots,
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    using namespace ascend_kernel;
    const uint32_t blockId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const __gm__ EchoLruTiling* tilingGm =
        reinterpret_cast<const __gm__ EchoLruTiling*>(tiling);
    const uint32_t R = tilingGm->numRequests;
    const uint32_t B = tilingGm->topkBufferB;
    const uint32_t topkDim = tilingGm->topkDim;
    const uint32_t maxModelLenP1 = tilingGm->maxModelLenP1;
    const uint32_t freeStackSize = tilingGm->freeStackSize;
    const uint32_t rowsPerCore = tilingGm->rowsPerCore;
    const uint32_t tailRows = tilingGm->tailRows;
    const uint32_t specPositions = tilingGm->specPositions;

    const uint32_t Pws = (specPositions > 4u) ? 4u : specPositions;
    const uint32_t specOn = tilingGm->specEnabled;

    const uint32_t rowStart = blockId * rowsPerCore;
    uint32_t rowEnd = rowStart + ((blockId == blockNum - 1u) ? tailRows : rowsPerCore);
    if (rowEnd > R) rowEnd = R;
    if (rowStart >= rowEnd) return;
    const uint32_t effTopk = (topkDim < B) ? topkDim : B;

    __gm__ int32_t* htdGm = reinterpret_cast<__gm__ int32_t*>(htd);
    __gm__ int32_t* dthGm = reinterpret_cast<__gm__ int32_t*>(dth);
    __gm__ int32_t* lruGm = reinterpret_cast<__gm__ int32_t*>(lru);
    __gm__ int32_t* freeGm = reinterpret_cast<__gm__ int32_t*>(free);
    __gm__ int32_t* availGm = reinterpret_cast<__gm__ int32_t*>(avail);
    __gm__ int32_t* fifoGm = reinterpret_cast<__gm__ int32_t*>(fifo);
    __gm__ int32_t* posGm = reinterpret_cast<__gm__ int32_t*>(pos);
    __gm__ int32_t* splGm = reinterpret_cast<__gm__ int32_t*>(spl);
    __gm__ uint8_t* resetMaskGm = reinterpret_cast<__gm__ uint8_t*>(resetMask);
    __gm__ uint8_t* freshGm = reinterpret_cast<__gm__ uint8_t*>(fresh);
    __gm__ int32_t* curSlotsGm = reinterpret_cast<__gm__ int32_t*>(curSlots);
    __gm__ int32_t* missCountGm = reinterpret_cast<__gm__ int32_t*>(missCount);
    __gm__ int32_t* missTokensGm = reinterpret_cast<__gm__ int32_t*>(missTokens);
    __gm__ int32_t* missSlotsGm = reinterpret_cast<__gm__ int32_t*>(missSlots);
    __gm__ int32_t* wsGm = reinterpret_cast<__gm__ int32_t*>(workspace);

#ifdef ECHO_WS_ABLATE_NOOP

    (void)htdGm; (void)dthGm; (void)lruGm; (void)freeGm; (void)availGm; (void)fifoGm;
    (void)posGm; (void)splGm; (void)resetMaskGm; (void)freshGm; (void)curSlotsGm;
    (void)missCountGm; (void)missTokensGm; (void)missSlotsGm; (void)wsGm;
    return;
#endif

#ifdef ECHO_WS_PHASE

    uint32_t wsPhaseOff = R * (B + 1u) + 5u * R + 3u * (R * specPositions) * topkDim;
    wsPhaseOff = (wsPhaseOff + 7u) & ~7u;
    wsPhaseOff += R * (3u * ((B + 31u) / 32u)) + R * 8u + blockNum * 2u;
    __gm__ int32_t* phaseGm = wsGm + wsPhaseOff + blockId * 16u;
    WsPhaseStamp(phaseGm, 8u);
#endif

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wsBuf;
    if (specPositions == 1u) {
        AscendC::TQue<AscendC::TPosition::VECIN, 1> dthQ;
        pipe.InitBuffer(dthQ, 1,
                        ECHO_WS_DTH_WORDS(topkDim, B, Pws) * sizeof(int32_t));
        pipe.InitBuffer(wsBuf, ECHO_WS_WS_ELEMS(topkDim, B, Pws) * sizeof(int32_t));
        AscendC::LocalTensor<int32_t> wsT = wsBuf.Get<int32_t>();
        __ubuf__ int32_t* ubBase =
            reinterpret_cast<__ubuf__ int32_t*>(wsT.GetPhyAddr());

        asc_vf_call<EchoWsFrontFrame>(
            dim3(ECHO_WS_VF_THREADS),
            htdGm, dthGm, lruGm, freeGm, availGm, fifoGm, splGm,
            resetMaskGm, freshGm, wsGm, ubBase,
            rowStart, rowEnd, B, maxModelLenP1, freeStackSize, R, specOn);
        WsVectorToScalar(pipe);

        AscendC::GlobalTensor<int32_t> dthGmT;
        dthGmT.SetGlobalBuffer(dthGm);
        const uint32_t dthCap = ECHO_WS_DTH_CAP(topkDim, B, Pws);
        for (uint32_t cs = rowStart; cs < rowEnd; cs += dthCap) {
            uint32_t ce = cs + dthCap;
            if (ce > rowEnd) ce = rowEnd;
            const uint32_t cw = (ce - cs) * (B + 1u);
            AscendC::LocalTensor<int32_t> dthT = dthQ.AllocTensor<int32_t>();

            AscendC::DataCopyExtParams cpD{1, static_cast<uint32_t>(cw * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padD{false, 0, 0, 0};
            AscendC::DataCopyPad(dthT, dthGmT[cs * (B + 1u)], cpD, padD);
            dthQ.EnQue(dthT);
            dthT = dthQ.DeQue<int32_t>();

            __ubuf__ int32_t* dthUb =
                reinterpret_cast<__ubuf__ int32_t*>(dthT.GetPhyAddr());
            asc_vf_call<EchoWsVfFrame>(
                dim3(ECHO_WS_VF_THREADS),
                htdGm, dthGm, lruGm, freeGm, availGm, fifoGm, posGm,
                freshGm, curSlotsGm, missCountGm, missTokensGm, missSlotsGm,
                wsGm, ubBase, dthUb,
                cs, ce, (cs == rowStart) ? 1u : 0u, blockId, B, topkDim,
                maxModelLenP1, freeStackSize, R, specOn);
            WsVectorToScalar(pipe);
            dthQ.FreeTensor(dthT);
        }
        return;
    }
    pipe.InitBuffer(wsBuf, ECHO_WS_ELEMS(topkDim, B, Pws) * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> wsT = wsBuf.Get<int32_t>();
    const uint64_t ubAddr = wsT.GetPhyAddr();
    __ubuf__ int32_t* ubBase = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
#ifdef ECHO_WS_PHASE

    WsPhaseStamp(phaseGm, 0u);
#endif

    asc_vf_call<EchoWsFrontFrame>(
        dim3(ECHO_WS_VF_THREADS),
        htdGm, dthGm, lruGm, freeGm, availGm, fifoGm, splGm,
        resetMaskGm, freshGm, wsGm, ubBase,
        rowStart, rowEnd, B, maxModelLenP1, freeStackSize, R, specOn);
    WsVectorToScalar(pipe);

#ifdef ECHO_WS_PHASE
    WsPhaseStamp(phaseGm, 1u);
#endif
    {
        AscendC::LocalTensor<float> sortF = wsBuf.Get<float>();
        AscendC::LocalTensor<int32_t> wsT32 = wsBuf.Get<int32_t>();

        const uint32_t Pg = specPositions;
        const uint32_t P = (Pg > 4u) ? 4u : Pg;
        const uint32_t C = Pg / P;
        const uint32_t validBit = (P == 2u) ? 0b0011u : 0b1111u;
        const uint32_t aOff = ECHO_WS_A_OFF;

        const uint32_t n = P * effTopk;
        const uint32_t chunkW = P * topkDim;
        const uint32_t flatDim = Pg * topkDim;
        const uint32_t numTokens = R * Pg;
        __ubuf__ int32_t* recUb = ubBase + ECHO_WS_B_OFF(topkDim, P);
        __ubuf__ int32_t* uniqueUb = ubBase + ECHO_WS_UNIQUE_OFF(topkDim, P);
        __ubuf__ int32_t* ownerUb = ubBase + ECHO_WS_OWNER_OFF(topkDim, P);

        __gm__ int32_t* missRun0Gm = wsGm + (R * (B + 1u) + 3u * R);
        __gm__ int32_t* wsMissRankGm = wsGm + (R * (B + 1u) + 5u * R + 2u * numTokens * topkDim);
        AscendC::GlobalTensor<int32_t> curSlotsGmT;
        curSlotsGmT.SetGlobalBuffer(curSlotsGm);
        AscendC::GlobalTensor<int32_t> missRankGmT;
        missRankGmT.SetGlobalBuffer(wsMissRankGm);

        AscendC::DataCopyExtParams cpCur{1, static_cast<uint32_t>(chunkW * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyExtParams cpMiss{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};

        asc_vf_call<EchoWsPackFrame>(
            dim3(ECHO_WS_VF_THREADS), posGm, curSlotsGm, ubBase,
            rowStart, topkDim, effTopk, P, 0u, Pg * topkDim, maxModelLenP1);
        WsVectorToScalar(pipe);

#ifdef ECHO_WS_PHASE
        WsPhaseStamp(phaseGm, 2u);
#endif
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            for (uint32_t chunk = 0u; chunk < C; ++chunk) {
                const uint32_t cbase = chunk * chunkW;
                const uint32_t isLastChunk = (chunk + 1u == C) ? 1u : 0u;
                const uint32_t hasNext = (isLastChunk != 0u) ? ((row + 1u < rowEnd) ? 1u : 0u) : 1u;
                const uint32_t nextRow = (isLastChunk != 0u) ? (row + 1u) : row;
                const uint32_t nextChunk = (isLastChunk != 0u) ? 0u : (chunk + 1u);

                AscendC::MrgSortSrcList<float> lists{
                    sortF[aOff], sortF[aOff + 2u * topkDim],
                    sortF[aOff + ((P > 2u) ? 4u * topkDim : 0u)],
                    sortF[aOff + ((P > 3u) ? 6u * topkDim : 0u)]};
                const uint16_t lens[4] = {static_cast<uint16_t>(effTopk),
                                          static_cast<uint16_t>(effTopk),
                                          static_cast<uint16_t>((P > 2u) ? effTopk : 0u),
                                          static_cast<uint16_t>((P > 3u) ? effTopk : 0u)};
                uint32_t used[4] = {0u, 0u, 0u, 0u};
#ifndef ECHO_WS_ABLATE_BUILD

                AscendC::MrgSort<float, false>(sortF[ECHO_WS_B_OFF(topkDim, P)], lists, lens, used, validBit, 1);
#endif
                WsVectorToScalar(pipe);
                WsScalarToVector(pipe);
#ifdef ECHO_WS_PHASE
                WsPhaseStamp(phaseGm, 3u);
#endif
#ifndef ECHO_WS_ABLATE_DEDUP

                asc_vf_call<WsDedupVf>(recUb, uniqueUb, ubBase, n);
                WsVectorToScalar(pipe);
#ifdef ECHO_WS_PHASE
                WsPhaseStamp(phaseGm, 4u);
#endif
#endif

                asc_vf_call<EchoWsJointFrame>(
                    dim3(ECHO_WS_VF_THREADS),
                    lruGm, dthGm, availGm, curSlotsGm, posGm, wsGm, ubBase,
                    row, rowEnd, blockId, B, topkDim, maxModelLenP1, R, P, 0u,
                    Pg, chunk, C, (chunk == 0u) ? 1u : 0u,
                    nextRow, nextChunk, hasNext, missRun0Gm);
                WsVectorToScalar(pipe);
#ifdef ECHO_WS_PHASE
                WsPhaseStamp(phaseGm, 5u);
#endif
#ifndef ECHO_WS_ABLATE_FINALIZE

                asc_vf_call<WsFinalizeVf>(recUb, uniqueUb, ownerUb, ubBase, n, topkDim,
                                          effTopk, P,
                                          (C == 1u) ? 0xFFFFFFFFu : (chunkW - 1u));

                WsVectorToMte3(pipe);

                AscendC::DataCopyPad(curSlotsGmT[row * flatDim + cbase], wsT32[ECHO_WS_A_OFF], cpCur);
                AscendC::DataCopyPad(missRankGmT[row * flatDim + cbase],
                                     wsT32[ECHO_WS_UNIQUE_OFF(topkDim, P)], cpMiss);

                WsMte3ToScalar(pipe);
#endif
#ifdef ECHO_WS_PHASE
                WsPhaseStamp(phaseGm, 6u);
#endif

                asc_vf_call<EchoWsJointFrame>(
                    dim3(ECHO_WS_VF_THREADS),
                    lruGm, dthGm, availGm, curSlotsGm, posGm, wsGm, ubBase,
                    row, rowEnd, blockId, B, topkDim, maxModelLenP1, R, P, 1u,
                    Pg, chunk, C, 0u,
                    nextRow, nextChunk, hasNext, missRun0Gm);
                WsVectorToScalar(pipe);

#ifdef ECHO_WS_PHASE
                WsPhaseStamp(phaseGm, 7u);
#endif
            }
        }
    }
}

namespace ascend_kernel {
constexpr uint32_t ECHO_WS_INIT_UB_BYTES = 512u;

// Frame body of the init kernel for one range of rows.
__simt_vf__ __aicore__ LAUNCH_BOUND(ECHO_WS_VF_THREADS) inline void EchoInitVfFrame(
    __gm__ int32_t* htdGm, __gm__ int32_t* dthGm, __gm__ int32_t* lruGm,
    __gm__ int32_t* freeGm, __gm__ int32_t* availGm, __gm__ int32_t* fifoGm,
    __gm__ uint8_t* freshGm, __gm__ uint8_t* initMaskGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t maxModelLenP1,
    uint32_t freeStackSize) {

    OpResetRows2(rowStart, rowEnd, maxModelLenP1, initMaskGm, htdGm);
    OpResetDthLru2(rowStart, rowEnd, B, initMaskGm, dthGm, lruGm);
    OpResetFree2(rowStart, rowEnd, B, freeStackSize, initMaskGm, freeGm);
    OpResetAvailFifo2(rowStart, rowEnd, B, initMaskGm, availGm, fifoGm);

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd;
         row += ECHO_WS_VF_THREADS) {
        if (initMaskGm[row] != 0u) freshGm[row] = 1;
    }
}
}

// Kernel entry of the init operator: builds the empty-pool state of the rows
// selected by init_mask.
extern "C" __global__ __aicore__ void echo_lru_init_kernel(
    GM_ADDR htd, GM_ADDR dth, GM_ADDR lru, GM_ADDR free, GM_ADDR avail, GM_ADDR fifo,
    GM_ADDR fresh, GM_ADDR initMask, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    using namespace ascend_kernel;
    const uint32_t blockId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const __gm__ EchoLruTiling* tilingGm =
        reinterpret_cast<const __gm__ EchoLruTiling*>(tiling);
    const uint32_t R = tilingGm->numRequests;
    const uint32_t B = tilingGm->topkBufferB;
    const uint32_t maxModelLenP1 = tilingGm->maxModelLenP1;
    const uint32_t freeStackSize = tilingGm->freeStackSize;
    const uint32_t rowsPerCore = tilingGm->rowsPerCore;
    const uint32_t tailRows = tilingGm->tailRows;
    const uint32_t rowStart = blockId * rowsPerCore;
    uint32_t rowEnd = rowStart + ((blockId == blockNum - 1u) ? tailRows : rowsPerCore);
    if (rowEnd > R) rowEnd = R;
    if (rowStart >= rowEnd) return;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stubBuf;
    pipe.InitBuffer(stubBuf, ECHO_WS_INIT_UB_BYTES);
    __ubuf__ int32_t* stubUb =
        reinterpret_cast<__ubuf__ int32_t*>(stubBuf.Get<int32_t>().GetPhyAddr());
    asc_vf_call<EchoInitVfFrame>(
        dim3(ECHO_WS_VF_THREADS),
        reinterpret_cast<__gm__ int32_t*>(htd), reinterpret_cast<__gm__ int32_t*>(dth),
        reinterpret_cast<__gm__ int32_t*>(lru), reinterpret_cast<__gm__ int32_t*>(free),
        reinterpret_cast<__gm__ int32_t*>(avail), reinterpret_cast<__gm__ int32_t*>(fifo),
        reinterpret_cast<__gm__ uint8_t*>(fresh), reinterpret_cast<__gm__ uint8_t*>(initMask),
        rowStart, rowEnd, B, maxModelLenP1, freeStackSize);
}
