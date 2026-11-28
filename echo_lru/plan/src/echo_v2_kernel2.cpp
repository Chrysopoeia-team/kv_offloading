// echo_v2_kernel2.cpp - kernel 2 of the Echo LRU plan operator: merge, admission
// and bookkeeping.
//
// It consumes what kernel 1 left in the workspace for a request row (masks,
// counts, the ordered miss list and the handoff flags) and:
//   1. compresses the lru order into the survivor and hit lists (GatherMask) and
//      reads the free-slot stack,
//   2. admits the misses: unique tokens in ascending order take slots from the
//      free stack, evicting the oldest unused slots when the pool is full,
//   3. rebuilds lru, updates htd / dth / free / avail / fifo and fills
//      current_slots for the admitted tokens,
//   4. pads miss_tokens / miss_slots with -1 and writes miss_count.
//
// Mixed translation unit with the same structure as kernel 1: a vector shell
// around one SIMT frame call per row.
//
// Reading order: echo_v2_kernel2 -> EchoV2VfFrame -> OpMissMergeAdmit2 (P > 1) or
// the P = 1 scan path -> OpFreePersist2 -> OpClassifyScatter2 -> OpAllocPop2 ->
// OpAllocAppend2 -> OpCompactTail2.

#include <cstdint>

#include "kernel_operator.h"

#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "echo_v2_kernel2_launch.h"
#include "echo_lru_tiling.h"

#ifndef KERNEL_TASK_TYPE_DEFAULT
// Fallback for toolchains that do not provide the task-type registration macro.
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace {

constexpr int32_t ECHO_V2_I32_MAX = 2147483647;

constexpr uint32_t ECHO_V2_VF_THREADS = 512u;

constexpr uint32_t ECHO_V2_SCAN_ELEMS = 128u;

constexpr uint32_t ECHO_V2_SCAN_TOTAL = ECHO_V2_SCAN_ELEMS;

#define ECHO_V2_UB_ELEMS(rowCount_, B_) \
    (ECHO_V2_SCAN_TOTAL + (rowCount_) * (B_))

constexpr uint32_t ECHO_V2_UB_LIMIT = 196608u;
constexpr uint32_t ECHO_V2_STG_CAP = 4096u;

#define ECHO_V2_UB_BASE_BYTES(B_, topkDim_, rowCount_) \
    ((B_) * 4u + 3u * (((B_) + 31u) / 32u) * 4u + 2u * (topkDim_) * 4u + \
     ECHO_V2_UB_ELEMS(rowCount_, B_) * 4u)

#define ECHO_V2_STG_OFF(rowCount_, B_) \
    ((ECHO_V2_UB_ELEMS(rowCount_, B_) + 7u) & ~7u)

#define ECHO_V2_STG_RESERVE_BYTES(rowCount_, B_) \
    ((ECHO_V2_STG_OFF(rowCount_, B_) - ECHO_V2_UB_ELEMS(rowCount_, B_) + \
      2u * (rowCount_) * ECHO_V2_STG_CAP) * 4u)

#define ECHO_V2_STG_ON(B_, topkDim_, rowCount_) \
    (ECHO_V2_UB_LIMIT >= \
     ECHO_V2_UB_BASE_BYTES(B_, topkDim_, rowCount_) + \
     ECHO_V2_STG_RESERVE_BYTES(rowCount_, B_))

// Warp/block cooperative scan used to rank the entries of the miss list.
__simt_callee__ __aicore__ inline int32_t PackedBlockScan2(int32_t packedValue,
                                                            __ubuf__ int32_t* warpTotals) {
    uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    uint32_t warp = static_cast<uint32_t>(threadIdx.x) >> 5;
    uint32_t warpCount = ECHO_V2_VF_THREADS >> 5;
    int32_t inclusive = packedValue;
    for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
        int32_t prior = asc_shfl_up(inclusive, offset, 32u);
        if (lane >= offset) inclusive += prior;
    }
    if (lane == 31u) warpTotals[warp] = inclusive;
    asc_syncthreads();

    int32_t base = 0;
    for (uint32_t w = 0; w < warp; ++w) base += warpTotals[w];
    return base + inclusive - packedValue;
}

// Block barrier with a D-cache flush between the steps of a row.
__simt_callee__ __aicore__ inline void SyncGridEntire2(__gm__ int32_t* wsBase) {
    asc_syncthreads();
    if (threadIdx.x == 0u) {
        asc_dcci_entire(reinterpret_cast<__gm__ void*>(wsBase));
    }
    asc_syncthreads();
}

// Event pair: vector writes done -> MTE3 may copy them out.
__aicore__ inline void EchoV2VectorToMte3(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(id);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(id);
}

// Event pair: scalar writes done -> MTE3 may copy them out.
__aicore__ inline void EchoV2ScalarToMte3(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::S_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(id);
}
// Event pair: MTE3 copy done -> vector side may read.
__aicore__ inline void EchoV2Mte3ToVector(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(id);
}
// Event pair: MTE3 copy done -> scalar side may continue.
__aicore__ inline void EchoV2Mte3ToScalar(AscendC::TPipe& pipe) {
    auto id = pipe.AllocEventID<AscendC::HardEvent::MTE3_S>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(id);
}

// Per-position emission of the P = 1 miss-admission path.
__simt_callee__ __aicore__ inline void MissScanUpdateEmit2(
    uint32_t flat, uint32_t rank, int32_t aslot, int32_t tok,
    uint32_t rowOff, uint32_t htdRowOff, uint32_t dthRowOff,
    uint32_t B, uint32_t maxModelLenP1,
    __gm__ int32_t* curSlotsGm, __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* htdGm, __gm__ int32_t* dthGm) {

    int32_t mposSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
    htdGm[htdRowOff + mposSafe] = aslot;
    int32_t s = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);

    dthGm[dthRowOff + s] = tok;
    curSlotsGm[rowOff + flat] = aslot;

    missTokensGm[rowOff + rank] = tok;
    missSlotsGm[rowOff + rank] = aslot;
}

// Rebuilds the lru order from the survivor and hit lists.
__simt_callee__ __aicore__ inline void OpClassifyScatter2(
    uint32_t row, uint32_t B,
    __gm__ int32_t* lruGm, __ubuf__ int32_t* survHitUb,
    __gm__ int32_t* countsGm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t cntOff = row * 8u;
    int32_t totalK = countsGm[cntOff + 4];
    int32_t totalHit = countsGm[cntOff + 1];
    int32_t rbBase = countsGm[cntOff + 3];

    if (rbBase < 0) rbBase = 0;
    if (rbBase > (int32_t)B) rbBase = (int32_t)B;
    if (totalK < 0) totalK = 0;
    if (totalK > rbBase) totalK = rbBase;
    if (totalHit < 0) totalHit = 0;
    if (totalHit > rbBase - totalK) totalHit = rbBase - totalK;

    __ubuf__ int32_t* hitUb = survHitUb + (uint32_t)totalK;
    uint32_t lruRowOff = row * ECHO_LRU_ROW_PITCH(B);
    for (uint32_t j = t; j < (uint32_t)totalK; j += ECHO_V2_VF_THREADS) {
        lruGm[lruRowOff + j] = survHitUb[j];
    }
    for (uint32_t j = t; j < (uint32_t)totalHit; j += ECHO_V2_VF_THREADS) {
        lruGm[lruRowOff + (uint32_t)totalK + j] = hitUb[j];
    }

}

// Cleans the victim slots and pushes them onto the free stack.
__simt_callee__ __aicore__ inline void OpFreePersist2(
    uint32_t row, uint32_t flatDim, uint32_t B, uint32_t maxModelLenP1, uint32_t freeStackSize,
    __gm__ int32_t* freeIdxGm, __gm__ int32_t* freeSizeGm, __gm__ int32_t* numMissesGm,
    __gm__ int32_t* countsGm,
    __gm__ int32_t* availGm, __gm__ int32_t* dthGm, __gm__ int32_t* htdGm, __gm__ int32_t* freeGm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t cols = B + 1u;
    int32_t fs = freeSizeGm[row];
    int32_t nm = numMissesGm[row];
    int32_t nv = countsGm[row * 8u + 2u];
    if (nv < 0) nv = 0;
    if (nv > (int32_t)flatDim) nv = (int32_t)flatDim;
    int32_t avail = availGm[row];
    uint32_t fiOff = row * flatDim;
    uint32_t dthRowOff = row * cols;
    uint32_t htdRowOff = row * maxModelLenP1;
    uint32_t freeRowOff = row * freeStackSize;
    for (uint32_t j = t; j < flatDim; j += ECHO_V2_VF_THREADS) {
        if ((int32_t)j >= nv) continue;
        int32_t slot = freeIdxGm[fiOff + j];
        int32_t s = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
        uint32_t off = dthRowOff + s;
        int32_t tok = dthGm[off];

        if ((int32_t)j < fs - nm) dthGm[off] = ECHO_V2_I32_MAX;
        if (tok != ECHO_V2_I32_MAX) {
            int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
            htdGm[htdRowOff + tokSafe] = ECHO_V2_I32_MAX;
        }
        int32_t writeIdx = avail + (int32_t)j;
        if (writeIdx < 0) writeIdx = 0;
        if (writeIdx >= (int32_t)freeStackSize) writeIdx = (int32_t)freeStackSize - 1;
        freeGm[freeRowOff + writeIdx] = slot;
    }
}

// Pops the slots of the admitted misses from the free stack.
__simt_callee__ __aicore__ inline void OpAllocPop2(
    uint32_t row, uint32_t flatDim, uint32_t freeStackSize,
    __gm__ int32_t* numMissesGm, __gm__ int32_t* countsGm, __gm__ int32_t* availGm,
    __gm__ int32_t* freeIdxGm, __gm__ int32_t* freeGm, __gm__ int32_t* rawSlotsGm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    int32_t nm = numMissesGm[row];
    int32_t avail = availGm[row];
    int32_t nv = countsGm[row * 8u + 2u];
    if (nv < 0) nv = 0;
    if (nv > (int32_t)flatDim) nv = (int32_t)flatDim;
    uint32_t rowOff = row * flatDim;
    uint32_t fiOff = row * flatDim;
    uint32_t freeRowOff = row * freeStackSize;

    for (uint32_t j = t; j < (uint32_t)nm && j < flatDim; j += ECHO_V2_VF_THREADS) {
        int32_t readIdx = avail + nv - (int32_t)nm + (int32_t)j;
        int32_t slot;
        if (readIdx >= avail) {
            int32_t fi = readIdx - avail;
            if (fi < 0) fi = 0;
            if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
            slot = freeIdxGm[fiOff + fi];
        } else {
            if (readIdx < 0) readIdx = 0;
            if (readIdx >= (int32_t)freeStackSize) readIdx = (int32_t)freeStackSize - 1;
            slot = freeGm[freeRowOff + readIdx];
        }
        rawSlotsGm[rowOff + j] = slot;
    }
}

// Appends the admitted slots to the lru tail and updates fifo and avail.
__simt_callee__ __aicore__ inline void OpAllocAppend2(
    uint32_t row, uint32_t B, uint32_t flatDim, uint32_t specOn, uint32_t admissions,
    __gm__ int32_t* lruGm, __gm__ int32_t* rawSlotsGm, __gm__ int32_t* countsGm,
    __gm__ int32_t* fifoGm, __gm__ int32_t* availGm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t lruPitch = ECHO_LRU_ROW_PITCH(B);
    uint32_t admU = admissions;
    int32_t nv = countsGm[row * 8u + 2u];
    uint32_t base = (uint32_t)countsGm[row * 8u + 3];
    uint32_t lruRowOff = row * lruPitch;
    uint32_t rawRowOff = row * flatDim;

    for (uint32_t j = t; j < admU; j += ECHO_V2_VF_THREADS) {
        int32_t a = rawSlotsGm[rawRowOff + j];
        int32_t ac = a < 0 ? 0 : (a > (int32_t)B ? (int32_t)B : a);
        lruGm[lruRowOff + base + j] = ac;
    }

    if (specOn != 0u || nv > (int32_t)admissions) {
        uint32_t occNew = base + admU;
        for (uint32_t p = occNew + t; p <= B; p += ECHO_V2_VF_THREADS) lruGm[lruRowOff + p] = 0;
    }

    if (t == 0u) {
        fifoGm[row] = fifoGm[row] + 2;
    } else if (t == 1u) {

        availGm[row] = availGm[row] + nv - (int32_t)admissions;
    }
}

// Invalidates dth for slots that were evicted but not reused by an admission.
__simt_callee__ __aicore__ inline void OpFreeUnusedDth2(
    uint32_t row, uint32_t flatDim, uint32_t B, uint32_t admissions,
    __gm__ int32_t* freeIdxGm, __gm__ int32_t* countsGm, __gm__ int32_t* dthGm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    int32_t nv = countsGm[row * 8u + 2u];
    if (nv < 0) nv = 0;
    if (nv > (int32_t)flatDim) nv = (int32_t)flatDim;
    int32_t unused = nv - (int32_t)admissions;
    if (unused <= 0) return;
    uint32_t fiOff = row * flatDim;
    uint32_t dthRowOff = row * (B + 1u);
    for (uint32_t j = t; j < (uint32_t)unused; j += ECHO_V2_VF_THREADS) {
        int32_t slot = freeIdxGm[fiOff + j];
        int32_t s = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
        dthGm[dthRowOff + s] = ECHO_V2_I32_MAX;
    }
}

__simt_callee__ __aicore__ inline int32_t MergePopEmit2(
    int32_t flat, int32_t tok, int32_t rank, int32_t anchor, int32_t availOld,
    uint32_t flatDim, uint32_t freeStackSize, uint32_t B, uint32_t maxModelLenP1,
    uint32_t rowOff, uint32_t fiOff, uint32_t rawRowOff, uint32_t freeRowOff,
    uint32_t htdRowOff, uint32_t dthRowOff,
    __gm__ int32_t* curSlotsGm, __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* rawSlotsGm, __gm__ int32_t* htdGm, __gm__ int32_t* dthGm,
    __gm__ int32_t* freeIdxGm, __gm__ int32_t* freeGm,
    __ubuf__ int32_t* stgTokRow, __ubuf__ int32_t* stgSlotRow, uint32_t W) {

    int32_t stackIndex = anchor + rank;
    int32_t aslot;
    if (stackIndex >= availOld) {
        int32_t fi = stackIndex - availOld;
        if (fi < 0) fi = 0;
        if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
        aslot = freeIdxGm[fiOff + fi];
    } else {
        if (stackIndex < 0) stackIndex = 0;
        if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
        aslot = freeGm[freeRowOff + stackIndex];
    }
    rawSlotsGm[rawRowOff + (uint32_t)rank] = aslot;

    int32_t mposSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
    htdGm[htdRowOff + mposSafe] = aslot;
    int32_t s = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);

    dthGm[dthRowOff + s] = tok;
    curSlotsGm[rowOff + (uint32_t)flat] = aslot;

    if ((uint32_t)rank < W) {
        stgTokRow[rank] = tok;
        stgSlotRow[rank] = aslot;
    } else {
        missTokensGm[rowOff + (uint32_t)rank] = tok;
        missSlotsGm[rowOff + (uint32_t)rank] = aslot;
    }
    return aslot;
}

// Two-run merge helper: binary-searches where the chunk of the merged miss
// list handled by one thread starts (index in run 0 and index in run 1).
__simt_callee__ __aicore__ inline void MissRunSplit(
    __gm__ int32_t* list, uint32_t rowOff, __gm__ int32_t* posGm,
    uint32_t nm0, uint32_t nm1, uint32_t run1Off, uint32_t q,
    uint32_t& iOut, uint32_t& jOut) {
    uint32_t lo = (q > nm1) ? (q - nm1) : 0u;
    uint32_t hi = (q < nm0) ? q : nm0;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1u) >> 1;
        uint32_t j = q - mid;
        bool ok;
        if (mid == 0u || j == nm1) {
            ok = true;
        } else {
            int32_t fa = list[rowOff + (mid - 1u)];
            int32_t fb = list[rowOff + run1Off + j];
            int32_t ta = posGm[rowOff + (uint32_t)fa];
            int32_t tb = posGm[rowOff + (uint32_t)fb];
            ok = (ta < tb) || (ta == tb && fa < fb);
        }
        if (ok) lo = mid; else hi = mid - 1u;
    }
    iOut = lo; jOut = q - lo;
}
// Two-run merge helper: returns the next element of the two sorted runs in
// merge order.
__simt_callee__ __aicore__ inline int32_t MissRunNext(
    __gm__ int32_t* list, uint32_t rowOff, __gm__ int32_t* posGm,
    uint32_t nm0, uint32_t nm1, uint32_t run1Off, uint32_t& i, uint32_t& j) {
    if (i >= nm0) return list[rowOff + run1Off + (j++)];
    if (j >= nm1) return list[rowOff + (i++)];
    int32_t fa = list[rowOff + i];
    int32_t fb = list[rowOff + run1Off + j];
    int32_t ta = posGm[rowOff + (uint32_t)fa];
    int32_t tb = posGm[rowOff + (uint32_t)fb];
    if (ta < tb || (ta == tb && fa < fb)) { ++i; return fa; }
    ++j; return fb;
}

// P > 1 miss chain in one callee: scans the ordered miss list (two sorted
// runs), counts raw and unique misses, chooses the eviction window and
// admits the unique tokens in ascending order.
__simt_callee__ __aicore__ inline uint32_t OpMissMergeAdmit2(
    uint32_t row, uint32_t flatDim, uint32_t B, uint32_t maxModelLenP1,
    uint32_t freeStackSize, int32_t availOld, int32_t availEv, uint32_t nm,
    uint32_t nm0, uint32_t run1Off,
    __gm__ int32_t* posGm, __gm__ int32_t* curSlotsGm,
    __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* rawSlotsGm, __gm__ int32_t* htdGm, __gm__ int32_t* dthGm,
    __gm__ int32_t* freeIdxGm, __gm__ int32_t* freeGm,
    __gm__ int32_t* wsMissRankGm, __ubuf__ int32_t* scanUb,
    __ubuf__ int32_t* stgTokRow, __ubuf__ int32_t* stgSlotRow, uint32_t W) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    const uint32_t Bm = ECHO_V2_VF_THREADS;
    const uint32_t warpCount = Bm >> 5;

    const uint32_t W2 = (nm + Bm - 1u) / Bm;
    const uint32_t qBase = t * W2;

    const uint32_t wEnd = (qBase + W2 < nm) ? (qBase + W2) : nm;
    const uint32_t rowOff = row * flatDim;

    const uint32_t nm1 = (nm0 < nm) ? (nm - nm0) : 0u;
    const uint32_t twoRuns = (nm1 != 0u) ? 1u : 0u;
    uint32_t mi = 0u, mj = 0u, miBase = 0u, mjBase = 0u;
    if (twoRuns != 0u) {
        MissRunSplit(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, qBase, mi, mj);
        miBase = mi; mjBase = mj;
    }
    const uint32_t fiOff = row * flatDim;
    const uint32_t rawRowOff = row * flatDim;
    const uint32_t freeRowOff = row * freeStackSize;
    const uint32_t htdRowOff = row * maxModelLenP1;
    const uint32_t dthRowOff = row * (B + 1u);

    int32_t bndTok = 0;
    int32_t bndHave = 0;
    if (t > 0u && qBase < nm) {
        int32_t bf;
        if (twoRuns != 0u) {

            if (mi == 0u) {
                bf = wsMissRankGm[rowOff + run1Off + (mj - 1u)];
            } else if (mj == 0u) {
                bf = wsMissRankGm[rowOff + (mi - 1u)];
            } else {
                int32_t fa = wsMissRankGm[rowOff + (mi - 1u)];
                int32_t fb = wsMissRankGm[rowOff + run1Off + (mj - 1u)];
                int32_t ta = posGm[rowOff + (uint32_t)fa];
                int32_t tb = posGm[rowOff + (uint32_t)fb];
                bf = ((ta > tb) || (ta == tb && fa > fb)) ? fa : fb;
            }
        } else {
            bf = wsMissRankGm[rowOff + (qBase - 1u)];
        }
        if (bf >= 0 && bf < (int32_t)flatDim) {
            bndTok = posGm[rowOff + (uint32_t)bf];
            bndHave = 1;
        }
    }

    const uint32_t chunks = (W2 + 3u) >> 2;
    int32_t f0 = -1, f1 = -1, f2 = -1, f3 = -1;
    int32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    int32_t cnt = 0;
    int32_t prevTok = bndTok;
    int32_t havePrev = bndHave;
    for (uint32_t c = 0u; c < chunks; ++c) {
        const uint32_t q0 = qBase + 4u * c;
        f0 = -1; f1 = -1; f2 = -1; f3 = -1;
        t0 = 0;  t1 = 0;  t2 = 0;  t3 = 0;
        if (twoRuns != 0u) {

            if (q0 < wEnd) {
                int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                if (fl >= 0 && fl < (int32_t)flatDim) { f0 = fl; t0 = posGm[rowOff + (uint32_t)fl]; }
            }
            if (q0 + 1u < wEnd) {
                int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                if (fl >= 0 && fl < (int32_t)flatDim) { f1 = fl; t1 = posGm[rowOff + (uint32_t)fl]; }
            }
            if (q0 + 2u < wEnd) {
                int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                if (fl >= 0 && fl < (int32_t)flatDim) { f2 = fl; t2 = posGm[rowOff + (uint32_t)fl]; }
            }
            if (q0 + 3u < wEnd) {
                int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                if (fl >= 0 && fl < (int32_t)flatDim) { f3 = fl; t3 = posGm[rowOff + (uint32_t)fl]; }
            }
        } else {
            if (q0 < wEnd) {
                int32_t fl = wsMissRankGm[rowOff + q0];
                if (fl >= 0 && fl < (int32_t)flatDim) { f0 = fl; t0 = posGm[rowOff + (uint32_t)fl]; }
            }
            if (q0 + 1u < wEnd) {
                int32_t fl = wsMissRankGm[rowOff + q0 + 1u];
                if (fl >= 0 && fl < (int32_t)flatDim) { f1 = fl; t1 = posGm[rowOff + (uint32_t)fl]; }
            }
            if (q0 + 2u < wEnd) {
                int32_t fl = wsMissRankGm[rowOff + q0 + 2u];
                if (fl >= 0 && fl < (int32_t)flatDim) { f2 = fl; t2 = posGm[rowOff + (uint32_t)fl]; }
            }
            if (q0 + 3u < wEnd) {
                int32_t fl = wsMissRankGm[rowOff + q0 + 3u];
                if (fl >= 0 && fl < (int32_t)flatDim) { f3 = fl; t3 = posGm[rowOff + (uint32_t)fl]; }
            }
        }

        if (f0 >= 0) {
            if (havePrev == 0 || (t0 - prevTok) != 0) cnt += 1;
            prevTok = t0; havePrev = 1;
        }
        if (f1 >= 0) {
            if (havePrev == 0 || (t1 - prevTok) != 0) cnt += 1;
            prevTok = t1; havePrev = 1;
        }
        if (f2 >= 0) {
            if (havePrev == 0 || (t2 - prevTok) != 0) cnt += 1;
            prevTok = t2; havePrev = 1;
        }
        if (f3 >= 0) {
            if (havePrev == 0 || (t3 - prevTok) != 0) cnt += 1;
            prevTok = t3; havePrev = 1;
        }
    }

    int32_t excl = PackedBlockScan2(cnt, scanUb);
    int32_t grand = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) grand += scanUb[w];
    uint32_t dedup = grand > 0 ? (uint32_t)grand : 0u;
    uint32_t admissions = (dedup > (uint32_t)availEv) ? (uint32_t)availEv : dedup;
    int32_t anchor = availEv - (int32_t)admissions;

    int32_t run = 0;
    int32_t curTok = 0;
    int32_t curSlot = ECHO_V2_I32_MAX;
    int32_t seenHead = 0;
    int32_t prevTok2 = bndTok;
    int32_t haveP2 = bndHave;

    if (twoRuns != 0u && chunks > 1u) { mi = miBase; mj = mjBase; }
    for (uint32_t c = 0u; c < chunks; ++c) {
        if (chunks > 1u) {

            const uint32_t q0 = qBase + 4u * c;
            f0 = -1; f1 = -1; f2 = -1; f3 = -1;
            t0 = 0;  t1 = 0;  t2 = 0;  t3 = 0;
            if (twoRuns != 0u) {
                if (q0 < wEnd) {
                    int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                    if (fl >= 0 && fl < (int32_t)flatDim) { f0 = fl; t0 = posGm[rowOff + (uint32_t)fl]; }
                }
                if (q0 + 1u < wEnd) {
                    int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                    if (fl >= 0 && fl < (int32_t)flatDim) { f1 = fl; t1 = posGm[rowOff + (uint32_t)fl]; }
                }
                if (q0 + 2u < wEnd) {
                    int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                    if (fl >= 0 && fl < (int32_t)flatDim) { f2 = fl; t2 = posGm[rowOff + (uint32_t)fl]; }
                }
                if (q0 + 3u < wEnd) {
                    int32_t fl = MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj);
                    if (fl >= 0 && fl < (int32_t)flatDim) { f3 = fl; t3 = posGm[rowOff + (uint32_t)fl]; }
                }
            } else {
                if (q0 < wEnd) {
                    int32_t fl = wsMissRankGm[rowOff + q0];
                    if (fl >= 0 && fl < (int32_t)flatDim) { f0 = fl; t0 = posGm[rowOff + (uint32_t)fl]; }
                }
                if (q0 + 1u < wEnd) {
                    int32_t fl = wsMissRankGm[rowOff + q0 + 1u];
                    if (fl >= 0 && fl < (int32_t)flatDim) { f1 = fl; t1 = posGm[rowOff + (uint32_t)fl]; }
                }
                if (q0 + 2u < wEnd) {
                    int32_t fl = wsMissRankGm[rowOff + q0 + 2u];
                    if (fl >= 0 && fl < (int32_t)flatDim) { f2 = fl; t2 = posGm[rowOff + (uint32_t)fl]; }
                }
                if (q0 + 3u < wEnd) {
                    int32_t fl = wsMissRankGm[rowOff + q0 + 3u];
                    if (fl >= 0 && fl < (int32_t)flatDim) { f3 = fl; t3 = posGm[rowOff + (uint32_t)fl]; }
                }
            }
        }

        if (f0 >= 0) {
            int32_t head = (haveP2 == 0 || (t0 - prevTok2) != 0) ? 1 : 0;
            prevTok2 = t0; haveP2 = 1;
            if (head != 0) {

                seenHead = 1;
                int32_t rank = excl + run;
                run += 1;
                curTok = t0;
                if ((uint32_t)rank < admissions) {

                    curSlot = MergePopEmit2(f0, t0, rank, anchor, availOld,
                                            flatDim, freeStackSize, B, maxModelLenP1,
                                            rowOff, fiOff, rawRowOff, freeRowOff,
                                            htdRowOff, dthRowOff,
                                            curSlotsGm, missTokensGm, missSlotsGm,
                                            rawSlotsGm, htdGm, dthGm, freeIdxGm, freeGm,
                        stgTokRow, stgSlotRow, W);
                } else {
                    curSlot = ECHO_V2_I32_MAX;
                }
            } else if (seenHead != 0 && curSlot != ECHO_V2_I32_MAX) {

                curSlotsGm[rowOff + (uint32_t)f0] = curSlot;
            }
        }

        if (f1 >= 0) {
            int32_t head = (haveP2 == 0 || (t1 - prevTok2) != 0) ? 1 : 0;
            prevTok2 = t1; haveP2 = 1;
            if (head != 0) {
                seenHead = 1;
                int32_t rank = excl + run;
                run += 1;
                curTok = t1;
                if ((uint32_t)rank < admissions) {
                    curSlot = MergePopEmit2(f1, t1, rank, anchor, availOld,
                                            flatDim, freeStackSize, B, maxModelLenP1,
                                            rowOff, fiOff, rawRowOff, freeRowOff,
                                            htdRowOff, dthRowOff,
                                            curSlotsGm, missTokensGm, missSlotsGm,
                                            rawSlotsGm, htdGm, dthGm, freeIdxGm, freeGm,
                        stgTokRow, stgSlotRow, W);
                } else {
                    curSlot = ECHO_V2_I32_MAX;
                }
            } else if (seenHead != 0 && curSlot != ECHO_V2_I32_MAX) {
                curSlotsGm[rowOff + (uint32_t)f1] = curSlot;
            }
        }

        if (f2 >= 0) {
            int32_t head = (haveP2 == 0 || (t2 - prevTok2) != 0) ? 1 : 0;
            prevTok2 = t2; haveP2 = 1;
            if (head != 0) {
                seenHead = 1;
                int32_t rank = excl + run;
                run += 1;
                curTok = t2;
                if ((uint32_t)rank < admissions) {
                    curSlot = MergePopEmit2(f2, t2, rank, anchor, availOld,
                                            flatDim, freeStackSize, B, maxModelLenP1,
                                            rowOff, fiOff, rawRowOff, freeRowOff,
                                            htdRowOff, dthRowOff,
                                            curSlotsGm, missTokensGm, missSlotsGm,
                                            rawSlotsGm, htdGm, dthGm, freeIdxGm, freeGm,
                        stgTokRow, stgSlotRow, W);
                } else {
                    curSlot = ECHO_V2_I32_MAX;
                }
            } else if (seenHead != 0 && curSlot != ECHO_V2_I32_MAX) {
                curSlotsGm[rowOff + (uint32_t)f2] = curSlot;
            }
        }

        if (f3 >= 0) {
            int32_t head = (haveP2 == 0 || (t3 - prevTok2) != 0) ? 1 : 0;
            prevTok2 = t3; haveP2 = 1;
            if (head != 0) {
                seenHead = 1;
                int32_t rank = excl + run;
                run += 1;
                curTok = t3;
                if ((uint32_t)rank < admissions) {
                    curSlot = MergePopEmit2(f3, t3, rank, anchor, availOld,
                                            flatDim, freeStackSize, B, maxModelLenP1,
                                            rowOff, fiOff, rawRowOff, freeRowOff,
                                            htdRowOff, dthRowOff,
                                            curSlotsGm, missTokensGm, missSlotsGm,
                                            rawSlotsGm, htdGm, dthGm, freeIdxGm, freeGm,
                        stgTokRow, stgSlotRow, W);
                } else {
                    curSlot = ECHO_V2_I32_MAX;
                }
            } else if (seenHead != 0 && curSlot != ECHO_V2_I32_MAX) {
                curSlotsGm[rowOff + (uint32_t)f3] = curSlot;
            }
        }
    }

    if (seenHead != 0 && curSlot != ECHO_V2_I32_MAX) {
        for (uint32_t q = wEnd; q < nm; ++q) {

            int32_t flat = (twoRuns != 0u)
                ? MissRunNext(wsMissRankGm, rowOff, posGm, nm0, nm1, run1Off, mi, mj)
                : wsMissRankGm[rowOff + q];
            if (flat < 0) break;
            if (flat >= (int32_t)flatDim) break;
            int32_t tok = posGm[rowOff + (uint32_t)flat];
            int32_t de = tok - curTok;
            if (de != 0) break;
            curSlotsGm[rowOff + (uint32_t)flat] = curSlot;
        }
    }
    return admissions;
}

// Fills the unused tail of miss_tokens / miss_slots with -1 and writes
// miss_count.
__simt_callee__ __aicore__ inline void OpCompactTail2(
    uint32_t row, uint32_t flatDim, uint32_t admissions,
    __gm__ int32_t* missCountGm, __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    const uint32_t rowOff = row * flatDim;
    for (uint32_t q = admissions + t; q < flatDim; q += ECHO_V2_VF_THREADS) {
        missTokensGm[rowOff + q] = -1;
        missSlotsGm[rowOff + q] = -1;
    }
    if (t == 0u) missCountGm[row] = (int32_t)admissions;
}

__simt_callee__ __aicore__ inline void OpMissScanUpdate2(
    uint32_t row, uint32_t r,
    uint32_t topkDim, uint32_t flatDim, uint32_t effTopk, uint32_t B, uint32_t maxModelLenP1,
    uint32_t freeStackSize,
    int32_t availOld, int32_t availEv, uint32_t admissions,
    __gm__ int32_t* curSlotsGm, __gm__ int32_t* posGm,
    __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* rawSlotsGm, __gm__ int32_t* htdGm, __gm__ int32_t* dthGm,
    __gm__ int32_t* freeIdxGm, __gm__ int32_t* freeGm,
    __ubuf__ int32_t* scanUb) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    const uint32_t Bm = ECHO_V2_VF_THREADS;
    const uint32_t warpCount = Bm >> 5;
    uint32_t rowOff = row * flatDim;
    uint32_t rawRowOff = row * flatDim;
    uint32_t fiOff = row * flatDim;
    uint32_t freeRowOff = row * freeStackSize;
    uint32_t htdRowOff = row * maxModelLenP1;
    uint32_t dthRowOff = row * (B + 1u);
    int32_t anchor = availEv - (int32_t)admissions;

    if (topkDim <= 2u * Bm) {

        uint32_t p0 = t;
        uint32_t p1 = t + Bm;

        int32_t ind0 = 0;
        bool miss0 = false;
        int32_t mpos0 = 0;
        if (p0 < topkDim) {
            miss0 = (p0 < effTopk) && (curSlotsGm[rowOff + p0] == ECHO_V2_I32_MAX);
            mpos0 = miss0 ? posGm[rowOff + p0] : 0;
            ind0 = miss0 ? 1 : 0;
        }
        int32_t ind1 = 0;
        bool miss1 = false;
        int32_t mpos1 = 0;
        if (p1 < topkDim) {
            miss1 = (p1 < effTopk) && (curSlotsGm[rowOff + p1] == ECHO_V2_I32_MAX);
            mpos1 = miss1 ? posGm[rowOff + p1] : 0;
            ind1 = miss1 ? 1 : 0;
        }
        __ubuf__ int32_t* buf = scanUb + (r & 1u) * 32u;
        int32_t excl = PackedBlockScan2(ind0 | (ind1 << 12), buf);
        int32_t e0 = excl & 0xFFF;
        int32_t e1 = (excl >> 12) & 0xFFF;
        int32_t grandPacked = 0;
        for (uint32_t w = 0u; w < warpCount; ++w) grandPacked += buf[w];
        int32_t total0 = grandPacked & 0xFFF;
        int32_t rank0 = ind0 ? e0 : 0;
        int32_t rank1 = ind1 ? (total0 + e1) : 0;

        if (p0 < topkDim && miss0) {
            uint32_t gfr = (uint32_t)rank0;
            if (gfr < admissions) {
                int32_t stackIndex = anchor + (int32_t)gfr;
                int32_t aslot;
                if (stackIndex >= availOld) {
                    int32_t fi = stackIndex - availOld;
                    if (fi < 0) fi = 0;
                    if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
                    aslot = freeIdxGm[fiOff + fi];
                } else {
                    if (stackIndex < 0) stackIndex = 0;
                    if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
                    aslot = freeGm[freeRowOff + stackIndex];
                }
                rawSlotsGm[rawRowOff + gfr] = aslot;
                MissScanUpdateEmit2(p0, gfr, aslot, mpos0, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                    curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
            }
        }
        if (p1 < topkDim && miss1) {
            uint32_t gfr = (uint32_t)rank1;
            if (gfr < admissions) {
                int32_t stackIndex = anchor + (int32_t)gfr;
                int32_t aslot;
                if (stackIndex >= availOld) {
                    int32_t fi = stackIndex - availOld;
                    if (fi < 0) fi = 0;
                    if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
                    aslot = freeIdxGm[fiOff + fi];
                } else {
                    if (stackIndex < 0) stackIndex = 0;
                    if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
                    aslot = freeGm[freeRowOff + stackIndex];
                }
                rawSlotsGm[rawRowOff + gfr] = aslot;
                MissScanUpdateEmit2(p1, gfr, aslot, mpos1, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                    curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
            }
        }
        return;
    }

    const uint32_t lane = t & 31u;
    const uint32_t warp = t >> 5;
    uint32_t p0 = t;
    uint32_t p1 = t + Bm;
    uint32_t p2 = t + 2u * Bm;
    uint32_t p3 = t + 3u * Bm;
    __ubuf__ int32_t* buf1 = scanUb + (r & 1u) * 64u;
    __ubuf__ int32_t* buf2 = buf1 + 32;

    int32_t ind0 = 0;
    bool miss0 = false;
    int32_t mpos0 = 0;
    if (p0 < topkDim) {
        miss0 = (p0 < effTopk) && (curSlotsGm[rowOff + p0] == ECHO_V2_I32_MAX);
        mpos0 = miss0 ? posGm[rowOff + p0] : 0;
        ind0 = miss0 ? 1 : 0;
    }
    int32_t ind1 = 0;
    bool miss1 = false;
    int32_t mpos1 = 0;
    if (p1 < topkDim) {
        miss1 = (p1 < effTopk) && (curSlotsGm[rowOff + p1] == ECHO_V2_I32_MAX);
        mpos1 = miss1 ? posGm[rowOff + p1] : 0;
        ind1 = miss1 ? 1 : 0;
    }
    int32_t ind2 = 0;
    bool miss2 = false;
    int32_t mpos2 = 0;
    if (p2 < topkDim) {
        miss2 = (p2 < effTopk) && (curSlotsGm[rowOff + p2] == ECHO_V2_I32_MAX);
        mpos2 = miss2 ? posGm[rowOff + p2] : 0;
        ind2 = miss2 ? 1 : 0;
    }
    int32_t ind3 = 0;
    bool miss3 = false;
    int32_t mpos3 = 0;
    if (p3 < topkDim) {
        miss3 = (p3 < effTopk) && (curSlotsGm[rowOff + p3] == ECHO_V2_I32_MAX);
        mpos3 = miss3 ? posGm[rowOff + p3] : 0;
        ind3 = miss3 ? 1 : 0;
    }

    int32_t packed1 = ind0 | (ind1 << 12);
    int32_t packed2 = ind2 | (ind3 << 12);
    int32_t incl1 = packed1;
    int32_t incl2 = packed2;
    for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
        int32_t prior1 = asc_shfl_up(incl1, offset, 32u);
        int32_t prior2 = asc_shfl_up(incl2, offset, 32u);
        if (lane >= offset) { incl1 += prior1; incl2 += prior2; }
    }
    if (lane == 31u) { buf1[warp] = incl1; buf2[warp] = incl2; }
    asc_syncthreads();
    int32_t base1 = 0, base2 = 0, grand1 = 0, grand2 = 0;
    for (uint32_t w = 0u; w < warpCount; ++w) {
        int32_t v1 = buf1[w], v2 = buf2[w];
        grand1 += v1; grand2 += v2;
        if (w < warp) { base1 += v1; base2 += v2; }
    }
    int32_t excl1 = base1 + incl1 - packed1;
    int32_t excl2 = base2 + incl2 - packed2;
    uint32_t e0 = (uint32_t)excl1 & 0xFFFu,  e1 = (uint32_t)excl1 >> 12;
    uint32_t e2 = (uint32_t)excl2 & 0xFFFu,  e3 = (uint32_t)excl2 >> 12;
    uint32_t t0  = (uint32_t)grand1 & 0xFFFu;
    uint32_t t01 = t0 + ((uint32_t)grand1 >> 12);
    uint32_t t2  = (uint32_t)grand2 & 0xFFFu;

    int32_t rank0 = ind0 ? (int32_t)e0 : 0;
    int32_t rank1 = ind1 ? (int32_t)(t0 + e1) : 0;
    int32_t rank2 = ind2 ? (int32_t)(t01 + e2) : 0;
    int32_t rank3 = ind3 ? (int32_t)(t01 + t2 + e3) : 0;

    if (p0 < topkDim && miss0) {
        uint32_t gfr = (uint32_t)rank0;
        if (gfr < admissions) {
            int32_t stackIndex = anchor + (int32_t)gfr;
            int32_t aslot;
            if (stackIndex >= availOld) {
                int32_t fi = stackIndex - availOld;
                if (fi < 0) fi = 0;
                if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
                aslot = freeIdxGm[fiOff + fi];
            } else {
                if (stackIndex < 0) stackIndex = 0;
                if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
                aslot = freeGm[freeRowOff + stackIndex];
            }
            rawSlotsGm[rawRowOff + gfr] = aslot;
            MissScanUpdateEmit2(p0, gfr, aslot, mpos0, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
        }
    }
    if (p1 < topkDim && miss1) {
        uint32_t gfr = (uint32_t)rank1;
        if (gfr < admissions) {
            int32_t stackIndex = anchor + (int32_t)gfr;
            int32_t aslot;
            if (stackIndex >= availOld) {
                int32_t fi = stackIndex - availOld;
                if (fi < 0) fi = 0;
                if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
                aslot = freeIdxGm[fiOff + fi];
            } else {
                if (stackIndex < 0) stackIndex = 0;
                if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
                aslot = freeGm[freeRowOff + stackIndex];
            }
            rawSlotsGm[rawRowOff + gfr] = aslot;
            MissScanUpdateEmit2(p1, gfr, aslot, mpos1, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
        }
    }
    if (p2 < topkDim && miss2) {
        uint32_t gfr = (uint32_t)rank2;
        if (gfr < admissions) {
            int32_t stackIndex = anchor + (int32_t)gfr;
            int32_t aslot;
            if (stackIndex >= availOld) {
                int32_t fi = stackIndex - availOld;
                if (fi < 0) fi = 0;
                if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
                aslot = freeIdxGm[fiOff + fi];
            } else {
                if (stackIndex < 0) stackIndex = 0;
                if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
                aslot = freeGm[freeRowOff + stackIndex];
            }
            rawSlotsGm[rawRowOff + gfr] = aslot;
            MissScanUpdateEmit2(p2, gfr, aslot, mpos2, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
        }
    }
    if (p3 < topkDim && miss3) {
        uint32_t gfr = (uint32_t)rank3;
        if (gfr < admissions) {
            int32_t stackIndex = anchor + (int32_t)gfr;
            int32_t aslot;
            if (stackIndex >= availOld) {
                int32_t fi = stackIndex - availOld;
                if (fi < 0) fi = 0;
                if (fi >= (int32_t)flatDim) fi = (int32_t)flatDim - 1;
                aslot = freeIdxGm[fiOff + fi];
            } else {
                if (stackIndex < 0) stackIndex = 0;
                if (stackIndex >= (int32_t)freeStackSize) stackIndex = (int32_t)freeStackSize - 1;
                aslot = freeGm[freeRowOff + stackIndex];
            }
            rawSlotsGm[rawRowOff + gfr] = aslot;
            MissScanUpdateEmit2(p3, gfr, aslot, mpos3, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
        }
    }
}

// Serial fallback of the P = 1 admission scan for topk values beyond the
// 4-element path.
__simt_callee__ __aicore__ inline void OpMissScanSerial2(
    uint32_t row, uint32_t topkDim, uint32_t flatDim, uint32_t effTopk, uint32_t B, uint32_t maxModelLenP1,
    __gm__ int32_t* curSlotsGm, __gm__ int32_t* posGm,
    __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* rawSlotsGm, __gm__ int32_t* htdGm, __gm__ int32_t* dthGm,
    __gm__ int32_t* wsMissRankGm, __gm__ int32_t* wsBase) {
    const uint32_t t = static_cast<uint32_t>(threadIdx.x);
    uint32_t priCols = B + 1u;

    uint32_t rowOff = row * flatDim;
    if (t == 0u) {

        int32_t rank = 0;
        for (uint32_t j = 0; j < topkDim; ++j) {
            bool miss = (j < effTopk) && (curSlotsGm[rowOff + j] == ECHO_V2_I32_MAX);
            wsMissRankGm[rowOff + j] = miss ? rank : 0;
            if (miss) rank += 1;
        }
    }

    SyncGridEntire2(wsBase);

    uint32_t rawRowOff = row * flatDim;
    uint32_t htdRowOff = row * maxModelLenP1;
    uint32_t dthRowOff = row * priCols;
    for (uint32_t j = t; j < topkDim; j += ECHO_V2_VF_THREADS) {
        bool miss = (j < effTopk) && (curSlotsGm[rowOff + j] == ECHO_V2_I32_MAX);
        int32_t rank = miss ? wsMissRankGm[rowOff + j] : 0;
        int32_t rk = rank < 0 ? 0 : rank;
        int32_t aslot = miss ? rawSlotsGm[rawRowOff + rk] : 0;
        int32_t tok = miss ? posGm[rowOff + j] : 0;
        if (miss) {
            MissScanUpdateEmit2(j, (uint32_t)rk, aslot, tok, rowOff, htdRowOff, dthRowOff, B, maxModelLenP1,
                                curSlotsGm, missTokensGm, missSlotsGm, htdGm, dthGm);
        }

    }
}

// Per-row frame: runs the admission chain and the bookkeeping steps of one
// request row.
__simt_vf__ __aicore__ LAUNCH_BOUND(ECHO_V2_VF_THREADS) inline void EchoV2VfFrame(
    __gm__ int32_t* lruGm, __gm__ int32_t* dthGm, __gm__ int32_t* htdGm,
    __gm__ int32_t* freeGm, __gm__ int32_t* availGm, __gm__ int32_t* fifoGm,
    __gm__ int32_t* posGm, __gm__ int32_t* curSlotsGm, __gm__ int32_t* missCountGm,
    __gm__ int32_t* missTokensGm, __gm__ int32_t* missSlotsGm,
    __gm__ int32_t* numMissesGm, __gm__ int32_t* freeSizeGm, __gm__ int32_t* freeIdxGm,
    __gm__ int32_t* rawSlotsGm, __gm__ int32_t* wsMissRankGm,
    __gm__ int32_t* missRun0Gm, __gm__ int32_t* countsGm,
    __gm__ int32_t* wsBaseGm, __ubuf__ int32_t* ubBase,
    uint32_t rowStart, uint32_t rowEnd, uint32_t B, uint32_t topkDim,
    uint32_t maxModelLenP1, uint32_t freeStackSize, uint32_t specPositions, uint32_t flags) {

    if ((flags & 2u) != 0u) return;

    const uint32_t effTopk = (topkDim < B) ? topkDim : B;
    const uint32_t flatDim = specPositions * topkDim;
    const uint32_t rowCount = rowEnd - rowStart;

    __ubuf__ int32_t* scanUb = ubBase;
    __ubuf__ int32_t* survUb = ubBase + ECHO_V2_SCAN_TOTAL;
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        const uint32_t r = row - rowStart;

        __ubuf__ int32_t* survRow = survUb + r * B;

#ifndef ECHO_V2_ABLATE_SCATTER

        OpClassifyScatter2(row, B, lruGm, survRow, countsGm);
#else
        (void)survRow;
#endif

        if ((flags & 1u) != 0u) {

#ifndef ECHO_V2_ABLATE_FREEP
            OpFreePersist2(row, flatDim, B, maxModelLenP1, freeStackSize,
                           freeIdxGm, freeSizeGm, numMissesGm, countsGm,
                           availGm, dthGm, htdGm, freeGm);
#endif
        }

        int32_t availOld = availGm[row];
        int32_t nvRow = countsGm[row * 8u + 2u];
        if (nvRow < 0) nvRow = 0;
        if (nvRow > (int32_t)flatDim) nvRow = (int32_t)flatDim;
        int32_t availEv = availOld + nvRow;
        int32_t nmRow = numMissesGm[row];
        if (nmRow < 0) nmRow = 0;
        uint32_t admissions;
        if (topkDim > 4u * ECHO_V2_VF_THREADS) {

            OpAllocPop2(row, flatDim, freeStackSize,
                        numMissesGm, countsGm, availGm, freeIdxGm, freeGm, rawSlotsGm);
            OpMissScanSerial2(row, topkDim, flatDim, effTopk, B, maxModelLenP1,
                              curSlotsGm, posGm, missTokensGm, missSlotsGm,
                              rawSlotsGm, htdGm, dthGm, wsMissRankGm, wsBaseGm);
            admissions = (uint32_t)nmRow;
        } else if (specPositions > 1u) {

            uint32_t nmU = (uint32_t)nmRow;
            if (nmU > flatDim) nmU = flatDim;

            uint32_t nm0 = nmU;
            uint32_t run1Off = 0u;
            if (specPositions > 4u) {
                int32_t n0 = missRun0Gm[row];
                if (n0 < 0) n0 = 0;
                if (n0 > (int32_t)nmU) n0 = (int32_t)nmU;
                nm0 = (uint32_t)n0;
                run1Off = flatDim >> 1;
            }
            admissions = 0u;
#ifndef ECHO_V2_ABLATE_MERGE
            if (nmU > 0u) {
                admissions = OpMissMergeAdmit2(row, flatDim, B, maxModelLenP1,
                                              freeStackSize, availOld, availEv, nmU,
                                              nm0, run1Off,
                                              posGm, curSlotsGm, missTokensGm, missSlotsGm,
                                              rawSlotsGm, htdGm, dthGm, freeIdxGm, freeGm,
                                              wsMissRankGm, scanUb,

                                              (__ubuf__ int32_t*)nullptr,
                                              (__ubuf__ int32_t*)nullptr, 0u);
            }
#else

            admissions = (nmU > (uint32_t)availEv) ? (uint32_t)availEv : nmU;
            (void)scanUb; (void)availOld;
#endif
        } else {

            uint32_t dedup = (uint32_t)nmRow;
            admissions = (dedup > (uint32_t)availEv) ? (uint32_t)availEv : dedup;

#ifndef ECHO_V2_ABLATE_MERGE
            OpMissScanUpdate2(row, r, topkDim, flatDim, effTopk, B, maxModelLenP1,
                              freeStackSize, availOld, availEv, admissions,
                              curSlotsGm, posGm, missTokensGm, missSlotsGm,
                              rawSlotsGm, htdGm, dthGm, freeIdxGm, freeGm, scanUb);
#endif
        }

#ifndef ECHO_V2_ABLATE_TAIL
        OpCompactTail2(row, flatDim, admissions, missCountGm, missTokensGm, missSlotsGm);

        OpFreeUnusedDth2(row, flatDim, B, admissions, freeIdxGm, countsGm, dthGm);
#else
        (void)missCountGm;
#endif

#ifndef ECHO_V2_ABLATE_SYNC
        SyncGridEntire2(wsBaseGm);
#endif

#ifndef ECHO_V2_ABLATE_ALLOC
        OpAllocAppend2(row, B, flatDim, (flags >> 2) & 1u, admissions,
                       lruGm, rawSlotsGm, countsGm, fifoGm, availGm);
#else
        (void)fifoGm;
#endif
    }
}

}

// Kernel entry of kernel 2 (decode step): reads the tiling struct, prepares
// the per-row scratch and calls the frame once per row.
extern "C" __global__ __aicore__ void echo_v2_kernel2(
    GM_ADDR lru, GM_ADDR dth, GM_ADDR htd, GM_ADDR free, GM_ADDR avail, GM_ADDR fifo,
    GM_ADDR pos, GM_ADDR curSlots, GM_ADDR missCount, GM_ADDR missTokens,
    GM_ADDR missSlots, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    const uint32_t blockId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());

    const __gm__ ascend_kernel::EchoLruTiling* tilingGm =
        reinterpret_cast<const __gm__ ascend_kernel::EchoLruTiling*>(tiling);

    const uint32_t R = tilingGm->numRequests;
    const uint32_t numTokens = tilingGm->numTokens;
    const uint32_t B = tilingGm->topkBufferB;
    const uint32_t topkDim = tilingGm->topkDim;
    const uint32_t maxModelLenP1 = tilingGm->maxModelLenP1;
    const uint32_t freeStackSize = tilingGm->freeStackSize;
    const uint32_t rowsPerCore = tilingGm->rowsPerCore;
    const uint32_t tailRows = tilingGm->tailRows;
    const uint32_t specOn = tilingGm->specEnabled;
    const uint32_t specPositions = tilingGm->specPositions;
    const uint32_t flatDim = specPositions * topkDim;

    __gm__ int32_t* ws = reinterpret_cast<__gm__ int32_t*>(workspace);
    uint32_t o = 0;
    o += R * (B + 1u);
    o += R;
    o += R;
    __gm__ int32_t* numMissesGm = ws + o; o += R;
    __gm__ int32_t* missRun0Gm = ws + o; o += R;
    __gm__ int32_t* freeIdxGm = ws + o; o += numTokens * topkDim;
    __gm__ int32_t* freeSizeGm = ws + o; o += R;
    __gm__ int32_t* rawSlotsGm = ws + o; o += numTokens * topkDim;
    __gm__ int32_t* wsMissRankGm = ws + o; o += numTokens * topkDim;
    o = (o + 7u) & ~7u;
    const uint32_t nWords = (B + 31u) / 32u;
    const uint32_t maskRowElems = 3u * nWords;
    __gm__ int32_t* masksGm = ws + o; o += R * maskRowElems;
    __gm__ int32_t* countsGm = ws + o; o += R * 8u;

    __gm__ int32_t* handoffGm = ws + o;

    const __gm__ int32_t* cntScalar = countsGm;

    bool allCold = true;
    for (uint32_t bn = 0; bn < blockNum; ++bn) {
        if (handoffGm[bn * 2u + 1u] == 0) {
            allCold = false;
            break;
        }
    }
    if (allCold) return;

#ifdef ECHO_V2_ABLATE_NOOP

    (void)numMissesGm; (void)freeIdxGm; (void)freeSizeGm; (void)rawSlotsGm;
    (void)wsMissRankGm; (void)masksGm; (void)cntScalar; (void)blockId;
    (void)maxModelLenP1; (void)freeStackSize; (void)rowsPerCore; (void)tailRows;
    (void)specOn; (void)flatDim;
    return;
#endif

    const uint32_t startRow = blockId * rowsPerCore;
    const uint32_t rowCount = (blockId == blockNum - 1u) ? tailRows : rowsPerCore;

    uint32_t rowEnd = startRow + rowCount;
    if (rowEnd > R) rowEnd = R;

    const uint32_t needFree = (handoffGm[blockId * 2u + 0u] != 0) ? 1u : 0u;
    const uint32_t coldSkip = (handoffGm[blockId * 2u + 1u] != 0) ? 1u : 0u;
    const uint32_t flags = needFree | (coldSkip << 1) | (specOn << 2);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> wQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> maskQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> fullQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
    pipe.InitBuffer(wQ, 1, B * sizeof(int32_t));
    pipe.InitBuffer(maskQ, 1, maskRowElems * sizeof(uint32_t));
    pipe.InitBuffer(fullQ, 2, topkDim * sizeof(int32_t));
    pipe.InitBuffer(ubBuf, ECHO_V2_UB_ELEMS(rowCount, B) * sizeof(int32_t));

    AscendC::GlobalTensor<int32_t> freeGmT;
    freeGmT.SetGlobalBuffer(freeIdxGm);
    AscendC::GlobalTensor<int32_t> lruGmT;
    lruGmT.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lru));
    AscendC::GlobalTensor<uint32_t> masksGmT;
    masksGmT.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(masksGm));

    AscendC::DataCopyExtParams cpW{1, static_cast<uint32_t>(B * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padW{false, 0, 0, 0};
    AscendC::DataCopyExtParams cpMask{1, static_cast<uint32_t>(maskRowElems * sizeof(uint32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> padMask{false, 0, 0, 0};
    AscendC::DataCopyExtParams cpFree{1, static_cast<uint32_t>(topkDim * sizeof(int32_t)), 0, 0, 0};

    AscendC::LocalTensor<int32_t> ubT = ubBuf.Get<int32_t>();

#ifndef ECHO_V2_ABLATE_EMIT

    for (uint32_t r = 0; r < rowCount; ++r) {
        const uint32_t row = startRow + r;
        if (row >= R) break;

        AscendC::LocalTensor<int32_t> wT = wQ.AllocTensor<int32_t>();
        AscendC::DataCopyPad(wT, lruGmT[row * ECHO_LRU_ROW_PITCH(B)], cpW, padW);
        wQ.EnQue(wT);
        wT = wQ.DeQue<int32_t>();
        AscendC::LocalTensor<uint32_t> maskT = maskQ.AllocTensor<uint32_t>();
        AscendC::DataCopyPad(maskT, masksGmT[row * maskRowElems], cpMask, padMask);
        maskQ.EnQue(maskT);
        maskT = maskQ.DeQue<uint32_t>();

        int32_t totalCand = cntScalar[row * 8u + 0];
        int32_t victims = cntScalar[row * 8u + 2];
        int32_t fsClamp = cntScalar[row * 8u + 5];

        int32_t rbBase = cntScalar[row * 8u + 3];
        int32_t totalK = cntScalar[row * 8u + 4];
        if (totalCand < 0) totalCand = 0;
        if (totalCand > (int32_t)B) totalCand = (int32_t)B;
        if (fsClamp < 0) fsClamp = 0;

        const int32_t fsWin = (flatDim <= static_cast<uint32_t>(B))
                                  ? static_cast<int32_t>(flatDim)
                                  : static_cast<int32_t>(topkDim);
        if (fsClamp > fsWin) fsClamp = fsWin;
        if (victims < 0) victims = 0;
        if (victims > totalCand) victims = totalCand;
        if (victims > fsClamp) victims = fsClamp;
        if (rbBase < 0) rbBase = 0;
        if (rbBase > (int32_t)B) rbBase = (int32_t)B;
        if (totalK < 0) totalK = 0;
        if (totalK > rbBase) totalK = rbBase;

        if (B > 8192u || (flatDim > topkDim && flatDim <= static_cast<uint32_t>(B)
                          && fsClamp > static_cast<int32_t>(topkDim))) {

            AscendC::LocalTensor<int32_t> routeT = ubT[ECHO_V2_SCAN_TOTAL];
            uint64_t rsvd0 = 0;
            AscendC::GatherMask(routeT, wT, maskT[0], true, B,
                                AscendC::GatherMaskParams(1, 1, 8, 0), rsvd0);
            AscendC::PipeBarrier<PIPE_V>();
            const int32_t winFlat = (int32_t)flatDim;

            for (int32_t j = victims; j < fsClamp; ++j) {
                routeT.SetValue((uint32_t)j, (int32_t)B + 1 + (j - totalCand));
            }

            if (fsClamp < winFlat) {
                const int32_t tailStart = fsClamp;
                const int32_t tailAlign = (fsClamp + 7) & ~7;
                for (int32_t j = tailStart; j < tailAlign && j < winFlat; ++j) {
                    routeT.SetValue((uint32_t)j, 0);
                }
                if (tailAlign < winFlat) {
                    AscendC::Duplicate(routeT[(uint32_t)tailAlign], 0,
                                       (uint32_t)(winFlat - tailAlign));
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            EchoV2VectorToMte3(pipe);
            EchoV2ScalarToMte3(pipe);
            AscendC::DataCopyExtParams cpFreeBig{
                1, static_cast<uint32_t>(winFlat * static_cast<int32_t>(sizeof(int32_t))), 0, 0, 0};
            AscendC::DataCopyPad(freeGmT[row * flatDim], routeT, cpFreeBig);

            EchoV2Mte3ToVector(pipe);
            AscendC::LocalTensor<int32_t> survT = ubT[ECHO_V2_SCAN_TOTAL + r * B];
            AscendC::LocalTensor<int32_t> hitT =
                ubT[ECHO_V2_SCAN_TOTAL + r * B + (uint32_t)totalK];
            uint64_t rsvd1 = 0;
            AscendC::GatherMask(survT, wT, maskT[nWords], true, B,
                                AscendC::GatherMaskParams(1, 1, 8, 0), rsvd1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::GatherMask(hitT, wT, maskT[2u * nWords], true, B,
                                AscendC::GatherMaskParams(1, 1, 8, 0), rsvd1);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
        AscendC::LocalTensor<int32_t> survT = ubT[ECHO_V2_SCAN_TOTAL + r * B];
        AscendC::LocalTensor<int32_t> hitT = ubT[ECHO_V2_SCAN_TOTAL + r * B + (uint32_t)totalK];

        AscendC::LocalTensor<int32_t> fullT = fullQ.AllocTensor<int32_t>();
        uint64_t rsvdCnt = 0;
        AscendC::GatherMask(fullT, wT, maskT[0], true, B,
                            AscendC::GatherMaskParams(1, 1, 8, 0), rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMask(survT, wT, maskT[nWords], true, B,
                            AscendC::GatherMaskParams(1, 1, 8, 0), rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMask(hitT, wT, maskT[2u * nWords], true, B,
                            AscendC::GatherMaskParams(1, 1, 8, 0), rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t j = victims; j < fsClamp; ++j) {
            fullT.SetValue(static_cast<uint32_t>(j), (int32_t)B + 1 + (j - totalCand));
        }
        if (fsClamp < static_cast<int32_t>(topkDim)) {
            const int32_t tailStart = fsClamp;
            const int32_t tailAlign = (fsClamp + 7) & ~7;
            for (int32_t j = tailStart; j < tailAlign && j < static_cast<int32_t>(topkDim); ++j) {
                fullT.SetValue(static_cast<uint32_t>(j), 0);
            }
            if (tailAlign < static_cast<int32_t>(topkDim)) {
                AscendC::Duplicate(fullT[static_cast<uint32_t>(tailAlign)], 0,
                                   static_cast<uint32_t>(static_cast<int32_t>(topkDim) - tailAlign));
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        fullQ.EnQue(fullT);
        fullT = fullQ.DeQue<int32_t>();
        AscendC::DataCopyPad(freeGmT[row * flatDim], fullT, cpFree);
        fullQ.FreeTensor(fullT);
        }

        wQ.FreeTensor(wT);
        maskQ.FreeTensor(maskT);
    }
#else

    (void)cpW; (void)padW; (void)cpMask; (void)padMask; (void)cpFree;
    (void)freeGmT; (void)lruGmT; (void)masksGmT; (void)cntScalar;
#endif

    const uint64_t ubAddr = ubT.GetPhyAddr();
    __ubuf__ int32_t* ubBase = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    asc_vf_call<EchoV2VfFrame>(
        dim3(ECHO_V2_VF_THREADS),
        reinterpret_cast<__gm__ int32_t*>(lru), reinterpret_cast<__gm__ int32_t*>(dth),
        reinterpret_cast<__gm__ int32_t*>(htd), reinterpret_cast<__gm__ int32_t*>(free),
        reinterpret_cast<__gm__ int32_t*>(avail), reinterpret_cast<__gm__ int32_t*>(fifo),
        reinterpret_cast<__gm__ int32_t*>(pos), reinterpret_cast<__gm__ int32_t*>(curSlots),
        reinterpret_cast<__gm__ int32_t*>(missCount), reinterpret_cast<__gm__ int32_t*>(missTokens),
        reinterpret_cast<__gm__ int32_t*>(missSlots),
        numMissesGm, freeSizeGm, freeIdxGm, rawSlotsGm, wsMissRankGm, missRun0Gm,
        countsGm, ws,
        ubBase, startRow, rowEnd, B, topkDim, maxModelLenP1, freeStackSize, specPositions, flags);
}

#include "platform/platform_ascendc.h"

namespace {

// Dynamic UB size of kernel 2 in bytes; single source with the launcher.
inline uint32_t echo_v2_ub_bytes(uint32_t B, uint32_t topkDim, uint32_t rowsPerCore) {
    const uint32_t maskRowElems = 3u * ((B + 31u) / 32u);
    return static_cast<uint32_t>(B * sizeof(int32_t)
                                 + maskRowElems * sizeof(uint32_t)
                                 + 2u * topkDim * sizeof(int32_t)
                                 + ECHO_V2_UB_ELEMS(rowsPerCore, B) * sizeof(int32_t));
}

}

// Host launcher of kernel 2.
void launch_echo_v2_kernel2(GM_ADDR lru, GM_ADDR dth, GM_ADDR htd, GM_ADDR free,
                            GM_ADDR avail, GM_ADDR fifo, GM_ADDR pos, GM_ADDR curSlots,
                            GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                            GM_ADDR workspace, GM_ADDR tiling,
                            uint32_t B, uint32_t topkDim, uint32_t rowsPerCore,
                            uint32_t blockNum, void* stream) {
    echo_v2_kernel2<<<blockNum, echo_v2_ub_bytes(B, topkDim, rowsPerCore), stream>>>(
        lru, dth, htd, free, avail, fifo, pos, curSlots, missCount, missTokens,
        missSlots, workspace, tiling);
}
