// echo_lru_kernel.h - kernel 1 of the unordered-TopK variant (v3.0.1 base):
// hit/miss lookup and mask emission.
//
// The TopK rows are not required to be sorted. For every request the P position
// rows are flattened into one query stream; a query-driven pass over htd
// classifies each query as hit or miss, and the row ends with the victim /
// survivor / hit masks that kernel 2 consumes. The misses are deduplicated and
// admitted in kernel 2.
//
// Pure SIMT translation unit (compiled with --enable-simt): class EchoLruKernel
// holds the step members that the entry kernels echo_lru_kernel and
// echo_lru_init_kernel call.

#include "echo_lru_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_warp_functions.h"

#ifdef ECHO_PERF_CLS_SUB
#ifndef ECHO_PERF_STEP
#error "ECHO_PERF_CLS_SUB requires ECHO_PERF_STEP (build.sh: ECHO_PERF_STEP=ON ECHO_PERF_CLS_SUB=ON)"
#endif
#endif

namespace ascend_kernel {

constexpr int32_t I32_MAX = 2147483647;

// Population count of a 32-bit word.
__aicore__ inline uint32_t Popcnt32(uint32_t x) {
    x = x - ((x >> 1u) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
    x = (x + (x >> 4u)) & 0x0F0F0F0Fu;
    x = (x + (x >> 8u)) & 0x00FF00FFu;
    return (x + (x >> 16u)) & 0x000000FFu;
}

// Returns mask with only its k lowest set bits kept.
__aicore__ inline uint32_t FirstKSetBits(uint32_t mask, uint32_t k) {
    uint32_t surv = mask;
    for (uint32_t r = 0u; r < k; ++r) surv &= surv - 1u;
    return mask & ~surv;
}

#ifdef ECHO_PERF_CLS_SUB

__attribute__((noinline)) __aicore__ void ClsSubStamp(int32_t* clsSubOut, uint32_t idx) {
    if (clsSubOut != nullptr && threadIdx.x == 0u) {
        clsSubOut[idx] = (int32_t)(uint32_t)clock();
    }
}
#endif

__attribute__((noinline)) __aicore__ void OpResetRows(
    uint32_t R, uint32_t maxModelLenP1, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* htdGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * maxModelLenP1;
        for (uint32_t col = threadIdx.x; col < maxModelLenP1; col += blockDim.x) {
            htdGm[rowOff + col] = (col == 0) ? 0 : I32_MAX;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpResetDthLru(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* dthGm, int32_t* lruGm) {

    uint32_t dthCols = B + 1;
    uint32_t lruPitch = ECHO_LRU_ROW_PITCH(B);
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        if (resetMaskGm[row] == 0) continue;
        uint32_t dthRowOff = row * dthCols;
        for (uint32_t col = threadIdx.x; col < dthCols; col += blockDim.x) {
            dthGm[dthRowOff + col] = I32_MAX;
        }

        uint32_t lruRowOff = row * lruPitch;
        for (uint32_t col = threadIdx.x; col < lruPitch; col += blockDim.x) {
            lruGm[lruRowOff + col] = 0;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpResetFree(
    uint32_t R, uint32_t B, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* freeGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        if (resetMaskGm[row] == 0) continue;
        uint32_t rowOff = row * freeStackSize;
        for (uint32_t col = threadIdx.x; col < freeStackSize; col += blockDim.x) {
            freeGm[rowOff + col] = (col < B) ? (int32_t)(B - col) : 0;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpResetAvailFifo(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    uint8_t* resetMaskGm, int32_t* availGm, int32_t* fifoGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            if (resetMaskGm[row] != 0) {
                availGm[row] = (int32_t)B;
                fifoGm[row] = 1;
            }
        }
    }
}

__attribute__((noinline)) __aicore__ void OpFreshClear(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows, uint8_t* freshGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0u) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            if (freshGm[row] != 0u) freshGm[row] = 0;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpColdScratchZero(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    uint32_t maskRowElems, int32_t* masksGm, int32_t* countsGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t mOff = row * maskRowElems;
        for (uint32_t i = threadIdx.x; i < maskRowElems; i += blockDim.x) {
            masksGm[mOff + i] = 0;
        }

        uint32_t cOff = row * 8u;
        for (uint32_t i = threadIdx.x; i < 8u; i += blockDim.x) {
            countsGm[cOff + i] = 0;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpInvalidateSpecSuffix(
    uint32_t R, uint32_t B, uint32_t maxModelLenP1, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* splGm, int32_t* dthGm,
    int32_t* htdGm, int32_t* wsCumsumGm) {
    uint32_t cols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t spl = splGm[row];
        uint32_t rowOff = row * cols;
        uint32_t htdRowOff = row * maxModelLenP1;
        for (uint32_t slot = threadIdx.x; slot < cols; slot += blockDim.x) {
            int32_t tok = dthGm[rowOff + slot];
            bool spec = (tok >= spl) && (tok != I32_MAX);
            wsCumsumGm[rowOff + slot] = spec ? 1 : 0;
            if (spec) {
                dthGm[rowOff + slot] = I32_MAX;
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                htdGm[htdRowOff + tokSafe] = I32_MAX;
            }
        }
    }
}

__attribute__((noinline)) __aicore__ void OpSpecCumsum(
    uint32_t R, uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* wsCumsumGm, int32_t* pushCountGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t acc = 0;
            for (uint32_t slot = 0; slot <= B; ++slot) {
                int32_t m = wsCumsumGm[row * (B + 1) + slot];
                acc += m;
                wsCumsumGm[row * (B + 1) + slot] = acc;
            }
            pushCountGm[row] = acc;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpSpecPushFree(
    uint32_t R, uint32_t B, uint32_t freeStackSize, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* wsCumsumGm, int32_t* availGm, int32_t* freeGm) {
    uint32_t cols = B + 1;
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        int32_t avail = availGm[row];
        uint32_t rowOff = row * cols;
        uint32_t freeRowOff = row * freeStackSize;
        for (uint32_t slot = threadIdx.x; slot < cols; slot += blockDim.x) {

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

__attribute__((noinline)) __aicore__ void OpSpecAvailAdd(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* availGm, int32_t* pushCountGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            availGm[row] = availGm[row] + pushCountGm[row];
        }
    }
}

__attribute__((noinline)) __aicore__ void OpHitMiss(
    uint32_t R, uint32_t topkDim, uint32_t effTopk, uint32_t maxModelLenP1,
    uint32_t B, uint32_t rowsPerCore, uint32_t tailRows,
    int32_t* htdGm, int32_t* posGm, int32_t* curSlotsGm,
    int32_t* numMissesGm, int32_t* availGm, int32_t* freeSizeGm,
    int32_t* warpScratch, uint32_t specPositions,
    int32_t* flagsUb = nullptr) {

    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    uint32_t t = threadIdx.x;
    uint32_t lane = t & 31u;
    uint32_t warpCount = blockDim.x >> 5;
    int32_t fsAny = 0;

    uint32_t flatDim = specPositions * topkDim;
    for (uint32_t r = 0; r < rowCount; ++r) {
        uint32_t row = startRow + r;
        if (row >= R) break;
        uint32_t rowOff = row * flatDim;
        uint32_t htdRowOff = row * maxModelLenP1;
        uint32_t localMiss = 0;

        if (flagsUb != nullptr) {
            for (uint32_t s = t; s <= B; s += blockDim.x) {
                flagsUb[s] = 0;
            }
            asc_syncthreads();
        }

        uint32_t pPos = 0u;
        uint32_t jj = t;
        for (uint32_t j = t; j < flatDim; j += blockDim.x) {

            if (jj >= effTopk) {
                curSlotsGm[rowOff + j] = 0;
            } else {
                int32_t tok = posGm[rowOff + j];
                int32_t tokSafe = tok < 0 ? 0 : (tok > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : tok);
                int32_t slot = htdGm[htdRowOff + tokSafe];

                curSlotsGm[rowOff + j] = slot;
                if (slot == I32_MAX) localMiss++;

                if (flagsUb != nullptr && slot != I32_MAX) {
                    int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
                    flagsUb[sv] = 1;
                }
            }

            jj += blockDim.x;
            if (jj >= topkDim) { jj -= topkDim; ++pPos; }
        }

        int32_t warpSum = asc_reduce_add((int32_t)localMiss);
        if (lane == 0u) warpScratch[(r & 1u) * 32u + (t >> 5)] = warpSum;
        asc_syncthreads();
        if (t == 0u) {
            int32_t nm = 0;
            for (uint32_t w = 0; w < warpCount; ++w) nm += warpScratch[(r & 1u) * 32u + w];
            numMissesGm[row] = nm;
            int32_t av = availGm[row];
            int32_t fs = nm - av;
            if (fs < 0) fs = 0;
            freeSizeGm[row] = fs;

            if (flagsUb != nullptr) {
                warpScratch[66] = fs;
                warpScratch[67] = av;
            }
            if (fs != 0) fsAny = 1;
        }
    }

    if (t == 0u) warpScratch[64] = fsAny;
}

__attribute__((noinline)) __aicore__ void OpFillZero(
    uint32_t R, uint32_t rowsPerCore, uint32_t tailRows, int32_t* outGm) {
    uint32_t startRow = blockIdx.x * rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
    if (threadIdx.x == 0) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            outGm[row] = 0;
        }
    }
}

__attribute__((noinline)) __aicore__ void OpClassifyMaskEmit(
    uint32_t B, uint32_t fsClamp, uint32_t occ,
    int32_t* lruGm, uint32_t lruRowOff,
    int32_t* curSlotsGm, uint32_t curRowOff,
    int32_t* dthGm, uint32_t dthRowOff,
    uint32_t effTopk, uint32_t topkDim, uint32_t specPositions, uint32_t specOn,

    int32_t* masksOut,
    int32_t* countsOut,

    int32_t* rbBaseOut,
    int32_t* flagsUb, int32_t* lruUb, int32_t* scanScratch,
    uint32_t flagsReady = 0u

#ifdef ECHO_PERF_CLS_SUB
    , int32_t* clsSubOut
#endif
    ) {
#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 0u);
#endif
    uint32_t t = threadIdx.x;
    uint32_t warpCount = blockDim.x >> 5;
    uint32_t cols = B + 1u;
    uint32_t nWords = (B + 31u) / 32u;
    uint32_t flatDim = specPositions * topkDim;

    if (occ == 0u && fsClamp == 0u) {
        for (uint32_t i = t; i < 3u * nWords; i += blockDim.x) {
            masksOut[i] = 0;
        }

        if (t == 0u) {
            rbBaseOut[0] = 0;
            for (uint32_t c = 0; c < 8u; ++c) countsOut[c] = 0;
        }
#ifdef ECHO_PERF_CLS_SUB
        ClsSubStamp(clsSubOut, 6u);
#endif
        return;
    }

#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 1u);
#endif

    if (flagsReady == 0u) {

    for (uint32_t s = t; s < cols; s += blockDim.x) {
        flagsUb[s] = 0;
    }

    asc_syncthreads();
    }

    if (specOn != 0u) {
        for (uint32_t p = t; p < occ; p += blockDim.x) {
            int32_t slot = lruGm[lruRowOff + p];
            int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
            if (dthGm[dthRowOff + sv] == I32_MAX) flagsUb[sv] = 2;
        }
    }

    if (flagsReady == 0u) {
    uint32_t pPos = 0u;
    uint32_t jj = t;
    for (uint32_t j = t; j < flatDim; j += blockDim.x) {
        if (jj < effTopk) {
            int32_t cs = curSlotsGm[curRowOff + j];
            if (cs != I32_MAX) {
                int32_t sv = cs < 0 ? 0 : (cs > (int32_t)B ? (int32_t)B : cs);
                flagsUb[sv] = 1;
            }
        }
        jj += blockDim.x;
        if (jj >= topkDim) { jj -= topkDim; ++pPos; }
    }
    }
#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 2u);
#endif
    asc_syncthreads();

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
            int32_t slot = lruGm[lruRowOff + p];
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

    uint32_t myCand = (wpw <= 32u) ? Popcnt32(candPk) : laneCand;
    uint32_t myHit = (wpw <= 32u) ? Popcnt32(hitPk) : laneHit;
    int32_t warpCand = asc_reduce_add((int32_t)myCand);
    int32_t warpHit = asc_reduce_add((int32_t)myHit);
    if (lane == 0u) scanScratch[warp] = warpCand | (warpHit << 16);
#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 3u);
#endif
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
#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 4u);
#endif

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
                int32_t slot = lruGm[lruRowOff + p];
                int32_t sv = slot < 0 ? 0 : (slot > (int32_t)B ? (int32_t)B : slot);
                int32_t v = flagsUb[sv];
                isHit = (v & 1) != 0;
                isCand = !isHit && ((v & 2) == 0);
            }
        }
        uint32_t candMask = asc_ballot(isCand ? 1 : 0);
        uint32_t hitMask = asc_ballot(isHit ? 1 : 0);
        uint32_t cnt = Popcnt32(candMask);

        uint32_t k = (fsClamp > C) ? ((fsClamp - C < cnt) ? (fsClamp - C) : cnt) : 0u;

        uint32_t victimMask = (k >= cnt) ? candMask : FirstKSetBits(candMask, k);
        uint32_t survMask = candMask & ~victimMask;
        if (lane == 0u) {
            masksOut[w] = (int32_t)victimMask;
            masksOut[nWords + w] = (int32_t)survMask;
            masksOut[2u * nWords + w] = (int32_t)hitMask;
        }
        C += cnt;
    }
#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 5u);
#endif

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
    asc_syncthreads();

#ifdef ECHO_PERF_CLS_SUB
    ClsSubStamp(clsSubOut, 6u);
#endif
}

// Steps of the lookup kernel. Each member works on the rows of one block.
class EchoLruKernel {
public:
    __aicore__ inline EchoLruKernel() {}

#ifdef ECHO_PERF_STEP
    __attribute__((noinline))
#endif
    __aicore__ inline void Init(int32_t* htd, int32_t* dth, int32_t* lru, int32_t* free,
                      int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
                      uint8_t* resetMask, uint8_t* fresh, int32_t* curSlots,
                      int32_t* missCount, int32_t* missTokens, int32_t* missSlots,
                      void* workspace, const EchoLruTiling& tiling,
                      int32_t* ubPool) {
        tiling_ = tiling;
        htdGm_ = htd;
        dthGm_ = dth;
        lruGm_ = lru;
        freeGm_ = free;
        availGm_ = avail;
        fifoGm_ = fifo;
        posGm_ = pos;
        splGm_ = spl;
        resetMaskGm_ = resetMask;
        freshGm_ = fresh;
        curSlotsGm_ = curSlots;
        missCountGm_    = missCount;
        missTokensGm_   = missTokens;
        missSlotsGm_    = missSlots;
        wsBase_ = workspace;

        int32_t* ws = (int32_t*)workspace;

        uint32_t R = tiling_.numRequests;
        uint32_t numTokens = tiling_.numTokens;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t o = 0;
        wsCumsumGm_   = ws + o; o += R * (B + 1);
        pushCountGm_  = ws + o; o += R;
        numHitsGm_    = ws + o; o += R;

        numMissesGm_  = ws + o; o += R;
        freeIdxGm_    = ws + o; o += numTokens * topkDim;

        freeSizeGm_   = ws + o; o += R;
        rawSlotsGm_   = ws + o; o += numTokens * topkDim;
        wsMissRankGm_ = ws + o; o += numTokens * topkDim;

        o = (o + 7u) & ~7u;
        maskRowElems_ = 3u * ((B + 31u) / 32u);
        masksGm_   = ws + o; o += R * maskRowElems_;
        countsGm_  = ws + o; o += R * 8u;

        handoffGm_ = ws + o; o += tiling_.blockNum * 2u;

        uint32_t bn = tiling_.blockNum;
        uint32_t elemOff = (o + 1u) & ~1u;

        stepAbsGm_ = nullptr;
#ifdef ECHO_PERF_STEP
        stepAbsGm_ = ws + elemOff;
        elemOff += bn * ECHO_PERF_NUM_STEPS;
#endif
        perfStepIdx_ = 0u;
#ifdef ECHO_PERF_CLS_SUB

        clsSubGm_ = ws + elemOff;
        elemOff += bn * ECHO_PERF_CLS_SUB_N;
#endif

        scanScratchUb_ = ubPool;

        uint32_t uo = 128u;
        uo = (uo + 255u) & ~255u;
        flagsUb_  = ubPool + uo; uo += B + 1u;
        uo = (uo + 255u) & ~255u;
        lruUb_    = ubPool + uo; uo += B + 1u;

        uo = (uo + 255u) & ~255u;

        if (uo + 2u * ECHO_LRU_THREAD_NUM + 64u <= ARGTOPK_UB_ELEMS) {
            rbScanUb_ = ubPool + uo; uo += 2u * ECHO_LRU_THREAD_NUM + 64u;
        } else {
            rbScanUb_ = ubPool;
        }
    }

    __aicore__ inline void ProcessA() {
        PerfEntry();

        uint32_t R = tiling_.numRequests;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t maxModelLenP1 = tiling_.maxModelLenP1;
        uint32_t effTopk = tiling_.effTopk;
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;

        if (tiling_.specEnabled) {
            OpFillZero(R, rowsPerCore, tailRows, pushCountGm_);
        }
        if (ECHO_V2_DEBUG_CUTOFF_V == 1u) return;

        bool anyReset = false;
        {
            uint32_t startRow = blockIdx.x * rowsPerCore;
            uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
            for (uint32_t r = 0; r < rowCount; ++r) {
                anyReset = anyReset || (resetMaskGm_[startRow + r] != 0);
            }
        }

        if (anyReset) OpResetRows(R, maxModelLenP1, rowsPerCore, tailRows, resetMaskGm_, htdGm_);
        if (anyReset) OpResetDthLru(R, B, rowsPerCore, tailRows, resetMaskGm_, dthGm_, lruGm_);
        if (anyReset) OpResetFree(R, B, tiling_.freeStackSize, rowsPerCore, tailRows, resetMaskGm_, freeGm_);
        if (anyReset) OpResetAvailFifo(R, B, rowsPerCore, tailRows, resetMaskGm_, availGm_, fifoGm_);
        if (anyReset) SyncGrid();
        if (ECHO_V2_DEBUG_CUTOFF_V == 2u) return;

        if (FreshDispatch()) {
            OpColdScratchZero(R, rowsPerCore, tailRows, maskRowElems_, masksGm_, countsGm_);
            if (threadIdx.x == 0u) {
                handoffGm_[blockIdx.x * 2u + 1u] = 1;
                handoffGm_[blockIdx.x * 2u + 0u] = 0;
            }
            if (ECHO_V2_DEBUG_CUTOFF_V == 3u) return;
            return;
        }
        if (ECHO_V2_DEBUG_CUTOFF_V == 3u) return;

        if (tiling_.specEnabled) RunSpecSteps();

        const bool fuseSingleRow = (rowsPerCore == 1u && tailRows == 1u &&
                                    gridDim.x == R && tiling_.specEnabled == 0u);
        OpHitMiss(R, topkDim, effTopk, maxModelLenP1, B, rowsPerCore, tailRows,
                  htdGm_, posGm_, curSlotsGm_, numMissesGm_, availGm_, freeSizeGm_,
                  scanScratchUb_, tiling_.specPositions,
                  fuseSingleRow ? flagsUb_ : nullptr);
        if (fuseSingleRow) {
            asc_syncthreads();
        } else {
            SyncGrid();
        }
        PerfEnd();

        if (ECHO_V2_DEBUG_CUTOFF_V == 4u) return;

        bool needFree = (scanScratchUb_[64] != 0);
#ifdef ECHO_PERF_CLS_SUB

        ClsSubStamp(clsSubGm_ + blockIdx.x * ECHO_PERF_CLS_SUB_N, 10u);
#endif
        if (fuseSingleRow) {

            const uint32_t row = blockIdx.x;
            const int32_t fs = scanScratchUb_[66];
            const int32_t av = scanScratchUb_[67];
            const uint32_t fsClamp = (fs <= 0) ? 0u
                                  : ((fs > (int32_t)topkDim) ? topkDim : (uint32_t)fs);
            const uint32_t occ = (av >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)av);
            int32_t* masksRow = masksGm_ + row * maskRowElems_;
            int32_t* countsRow = countsGm_ + row * 8u;
            OpClassifyMaskEmit(B, fsClamp, occ, lruGm_, row * ECHO_LRU_ROW_PITCH(B),
                               curSlotsGm_, row * (tiling_.specPositions * topkDim),
                               dthGm_, row * (B + 1u),
                               effTopk, topkDim, tiling_.specPositions, tiling_.specEnabled,
                               masksRow, countsRow, numHitsGm_ + row,
                               flagsUb_, lruUb_, rbScanUb_, 1u
#ifdef ECHO_PERF_CLS_SUB
                               , nullptr
#endif
                               );
        } else {
            ClassifyMaskEmitAllRows();
        }
        PerfEnd();

        if (threadIdx.x == 0u) {
            handoffGm_[blockIdx.x * 2u + 0u] = needFree ? 1 : 0;
            handoffGm_[blockIdx.x * 2u + 1u] = 0;
        }
        if (ECHO_V2_DEBUG_CUTOFF_V == 5u) return;

    }

private:

    __aicore__ inline void SyncGrid() {
        asc_syncthreads();
        if (threadIdx.x == 0) asc_dcci_entire(wsBase_);
        asc_syncthreads();
    }

    __aicore__ inline void PerfEntry() {
#ifdef ECHO_PERF_STEP
        if (threadIdx.x == 0 && stepAbsGm_ != nullptr) {
            stepAbsGm_[blockIdx.x * ECHO_PERF_NUM_STEPS] = (int32_t)(uint32_t)clock();
        }
#else
        (void)0;
#endif
    }

    __aicore__ inline void PerfEnd() {
#ifdef ECHO_PERF_STEP
#ifdef ECHO_PERF_DRAIN
        SyncGrid();
#endif
        if (threadIdx.x == 0) {
            ++perfStepIdx_;
            if (stepAbsGm_ != nullptr && perfStepIdx_ < ECHO_PERF_NUM_STEPS) {
                stepAbsGm_[blockIdx.x * ECHO_PERF_NUM_STEPS + perfStepIdx_] =
                    (int32_t)(uint32_t)clock();
            }
        }
#else
        (void)0;
#endif
    }

    __attribute__((noinline)) __aicore__ bool FreshDispatch() {
        uint32_t R = tiling_.numRequests;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t effTopk = tiling_.effTopk;
        uint32_t maxModelLenP1 = tiling_.maxModelLenP1;
        uint32_t freeStackSize = tiling_.freeStackSize;
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;
        uint32_t t = threadIdx.x;
        uint32_t priCols = B + 1u;
        uint32_t startRow = blockIdx.x * rowsPerCore;
        uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;

        uint32_t specPositions = tiling_.specPositions;

        bool anyFresh = false;
        bool allFresh = true;
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            uint8_t fv = freshGm_[row];
            anyFresh = anyFresh || (fv != 0u);
            allFresh = allFresh && (fv != 0u);
        }
        if (allFresh && specPositions == 1u && tiling_.specEnabled == 0u) {

            for (uint32_t r = 0; r < rowCount; ++r) {
                uint32_t row = startRow + r;
                if (row >= R) break;
                if (freshGm_[row] == 0u) continue;
                int32_t availOld = availGm_[row];

                uint32_t rowOff = row * topkDim;
                uint32_t htdRowOff = row * maxModelLenP1;
                uint32_t dthRowOff = row * priCols;

                uint32_t lruRowOff = row * ECHO_LRU_ROW_PITCH(B);
                uint32_t freeRowOff = row * freeStackSize;
                for (uint32_t j = t; j < topkDim; j += blockDim.x) {
                    if (j >= effTopk) {

                        curSlotsGm_[rowOff + j] = 0;
                        missTokensGm_[rowOff + j] = -1;
                        missSlotsGm_[rowOff + j] = -1;
                        continue;
                    }

                    int32_t readIdx = availOld - (int32_t)effTopk + (int32_t)j;
                    if (readIdx < 0) readIdx = 0;
                    if (readIdx >= (int32_t)freeStackSize) readIdx = (int32_t)freeStackSize - 1;
                    int32_t aslot = freeGm_[freeRowOff + readIdx];
                    int32_t mpos = posGm_[rowOff + j];
                    int32_t mposSafe = mpos < 0 ? 0 : (mpos > (int32_t)(maxModelLenP1 - 1) ? (int32_t)(maxModelLenP1 - 1) : mpos);
                    curSlotsGm_[rowOff + j] = aslot;

                    missTokensGm_[rowOff + j] = mpos;
                    missSlotsGm_[rowOff + j] = aslot;
                    htdGm_[htdRowOff + mposSafe] = aslot;
                    int32_t s = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);
                    dthGm_[dthRowOff + s] = mpos;
                    int32_t ac = aslot < 0 ? 0 : (aslot > (int32_t)B ? (int32_t)B : aslot);
                    lruGm_[lruRowOff + j] = ac;
                }
                asc_syncthreads();
                if (t == 0u) {
                    availGm_[row] = availOld - (int32_t)effTopk;
                    fifoGm_[row] = fifoGm_[row] + 2;
                    freshGm_[row] = 0;
                    missCountGm_[row] = (int32_t)effTopk;
                }
            }

            return true;
        }

        if (anyFresh) OpFreshClear(R, rowsPerCore, tailRows, freshGm_);

        return false;
    }

    __attribute__((noinline)) __aicore__ void RunSpecSteps() {
        uint32_t R = tiling_.numRequests;
        uint32_t B = tiling_.topkBufferB;
        uint32_t maxModelLenP1 = tiling_.maxModelLenP1;
        uint32_t freeStackSize = tiling_.freeStackSize;
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;
        OpInvalidateSpecSuffix(R, B, maxModelLenP1, rowsPerCore, tailRows, splGm_, dthGm_, htdGm_, wsCumsumGm_); PerfEnd();
        SyncGrid(); PerfEnd();
        OpSpecCumsum(R, B, rowsPerCore, tailRows, wsCumsumGm_, pushCountGm_); PerfEnd();
        SyncGrid(); PerfEnd();
        OpSpecPushFree(R, B, freeStackSize, rowsPerCore, tailRows, wsCumsumGm_, availGm_, freeGm_); PerfEnd();
        SyncGrid(); PerfEnd();
        OpSpecAvailAdd(R, rowsPerCore, tailRows, availGm_, pushCountGm_); PerfEnd();
        SyncGrid(); PerfEnd();
    }

    __aicore__ inline void ClassifyMaskEmitAllRows() {
#ifdef ECHO_PERF_CLS_SUB

        ClsSubStamp(clsSubGm_ + blockIdx.x * ECHO_PERF_CLS_SUB_N, 11u);
#endif
        uint32_t rowsPerCore = tiling_.rowsPerCore;
        uint32_t tailRows = tiling_.tailRows;
        uint32_t startRow = blockIdx.x * rowsPerCore;
        uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tailRows : rowsPerCore;
        uint32_t B = tiling_.topkBufferB;
        uint32_t topkDim = tiling_.topkDim;
        uint32_t R = tiling_.numRequests;
        uint32_t effTopk = tiling_.effTopk;
        uint32_t specPositions = tiling_.specPositions;
        uint32_t flatDim = specPositions * topkDim;
        uint32_t specOn = tiling_.specEnabled;
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t row = startRow + r;
            if (row >= R) break;
            int32_t fs = freeSizeGm_[row];
            uint32_t fsClamp = (fs <= 0) ? 0u
                             : ((fs > (int32_t)topkDim) ? topkDim : (uint32_t)fs);
#ifdef ECHO_PERF_CLS_SUB

            int32_t* clsSub = (row == startRow)
                                  ? (clsSubGm_ + blockIdx.x * ECHO_PERF_CLS_SUB_N)
                                  : nullptr;
            ClsSubStamp(clsSub, 12u);

#endif
            int32_t avail = availGm_[row];
            uint32_t occ = (avail >= (int32_t)B) ? 0u : ((uint32_t)B - (uint32_t)avail);

            int32_t* masksRow = masksGm_ + row * maskRowElems_;
            int32_t* countsRow = countsGm_ + row * 8u;
#ifdef ECHO_PERF_CLS_SUB
            ClsSubStamp(clsSub, 13u);

#endif
            OpClassifyMaskEmit(B, fsClamp, occ,
                               lruGm_, row * ECHO_LRU_ROW_PITCH(B),
                               curSlotsGm_, row * flatDim,
                               dthGm_, row * (B + 1u),
                               effTopk, topkDim, specPositions, specOn,
                               masksRow, countsRow,
                               numHitsGm_ + row,
                               flagsUb_, lruUb_, rbScanUb_
#ifdef ECHO_PERF_CLS_SUB
                               , clsSub
#endif
                               );
#ifdef ECHO_PERF_CLS_SUB
            ClsSubStamp(clsSub, 14u);

#endif
        }
#ifdef ECHO_PERF_CLS_SUB
        ClsSubStamp(clsSubGm_ + blockIdx.x * ECHO_PERF_CLS_SUB_N, 15u);

#endif
    }

    EchoLruTiling tiling_;
    int32_t* htdGm_;
    int32_t* dthGm_;
    int32_t* lruGm_;
    int32_t* freeGm_;
    int32_t* availGm_;
    int32_t* fifoGm_;
    int32_t* posGm_;
    int32_t* splGm_;
    uint8_t* resetMaskGm_;
    uint8_t* freshGm_;
    int32_t* curSlotsGm_;
    int32_t* wsCumsumGm_;
    int32_t* pushCountGm_;
    int32_t* numHitsGm_;
    int32_t* numMissesGm_;
    int32_t* freeIdxGm_;
    int32_t* freeSizeGm_;
    int32_t* rawSlotsGm_;
    int32_t* wsMissRankGm_;

    int32_t* masksGm_;
    int32_t* countsGm_;

    int32_t* handoffGm_;
    uint32_t maskRowElems_;
    int32_t* missCountGm_;
    int32_t* missTokensGm_;
    int32_t* missSlotsGm_;
    int32_t* scanScratchUb_;
    int32_t* rbScanUb_;

    int32_t* flagsUb_;
    int32_t* lruUb_;

    void* wsBase_;
    int32_t* stepAbsGm_;
    uint32_t perfStepIdx_;
#ifdef ECHO_PERF_CLS_SUB
    int32_t* clsSubGm_;
#endif

};

extern "C" __global__ __launch_bounds__(ECHO_LRU_THREAD_NUM) void echo_lru_kernel(
    int32_t* htd, int32_t* dth, int32_t* lru, int32_t* free,
    int32_t* avail, int32_t* fifo, int32_t* pos, int32_t* spl,
    uint8_t* resetMask, uint8_t* fresh, int32_t* curSlots,
    int32_t* missCount, int32_t* missTokens, int32_t* missSlots,
    void* workspace, EchoLruTiling* tilingBuf) {
    EchoLruTiling tiling;
    tiling.blockNum       = tilingBuf->blockNum;
    tiling.rowsPerCore    = tilingBuf->rowsPerCore;
    tiling.tailRows       = tilingBuf->tailRows;
    tiling.numRequests    = tilingBuf->numRequests;
    tiling.numTokens      = tilingBuf->numTokens;
    tiling.topkBufferB    = tilingBuf->topkBufferB;
    tiling.freeStackSize  = tilingBuf->freeStackSize;
    tiling.maxModelLenP1  = tilingBuf->maxModelLenP1;
    tiling.topkDim        = tilingBuf->topkDim;
    tiling.effTopk        = tilingBuf->effTopk;
    tiling.specPositions  = tilingBuf->specPositions;
    tiling.blockSize      = tilingBuf->blockSize;
    tiling.blockTableCols = tilingBuf->blockTableCols;
    tiling.kDim           = tilingBuf->kDim;
    tiling.vDim           = tilingBuf->vDim;
    tiling.kvHeadNum      = tilingBuf->kvHeadNum;
    tiling.specEnabled    = tilingBuf->specEnabled;
    tiling.argtopkDebugRow = tilingBuf->argtopkDebugRow;
    tiling.argtopkGroupThreads = tilingBuf->argtopkGroupThreads;

    __ubuf__ int32_t argtopkUb[ARGTOPK_UB_ELEMS];
    EchoLruKernel op;
    op.Init(htd, dth, lru, free, avail, fifo, pos, spl, resetMask, fresh, curSlots,
            missCount, missTokens, missSlots, workspace, tiling, argtopkUb);

    op.ProcessA();
}

extern "C" __global__ __launch_bounds__(ECHO_LRU_THREAD_NUM) void echo_lru_init_kernel(
    int32_t* htd, int32_t* dth, int32_t* lru, int32_t* free,
    int32_t* avail, int32_t* fifo, uint8_t* fresh, uint8_t* initMask,
    EchoLruTiling* tilingBuf) {
    EchoLruTiling tiling;
    tiling.blockNum       = tilingBuf->blockNum;
    tiling.rowsPerCore    = tilingBuf->rowsPerCore;
    tiling.tailRows       = tilingBuf->tailRows;
    tiling.numRequests    = tilingBuf->numRequests;
    tiling.numTokens      = tilingBuf->numTokens;
    tiling.topkBufferB    = tilingBuf->topkBufferB;
    tiling.freeStackSize  = tilingBuf->freeStackSize;
    tiling.maxModelLenP1  = tilingBuf->maxModelLenP1;

    OpResetRows(tiling.numRequests, tiling.maxModelLenP1, tiling.rowsPerCore, tiling.tailRows,
                initMask, htd);
    OpResetDthLru(tiling.numRequests, tiling.topkBufferB, tiling.rowsPerCore, tiling.tailRows,
                  initMask, dth, lru);
    OpResetFree(tiling.numRequests, tiling.topkBufferB, tiling.freeStackSize, tiling.rowsPerCore,
                tiling.tailRows, initMask, free);
    OpResetAvailFifo(tiling.numRequests, tiling.topkBufferB, tiling.rowsPerCore, tiling.tailRows,
                     initMask, avail, fifo);

    uint32_t startRow = blockIdx.x * tiling.rowsPerCore;
    uint32_t rowCount = (blockIdx.x == gridDim.x - 1) ? tiling.tailRows : tiling.rowsPerCore;
    uint32_t rowEnd = startRow + rowCount;
    if (rowEnd > tiling.numRequests) rowEnd = tiling.numRequests;
    for (uint32_t row = startRow + threadIdx.x; row < rowEnd; row += blockDim.x) {
        if (initMask[row] != 0u) fresh[row] = 1;
    }
}

}
