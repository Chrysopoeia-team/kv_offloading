/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MulanPSL-2.0 AND Apache-2.0
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Sparse KV Plan with ordered queries and exact resident lookup.
 * Sources: op_test_kvoffload_simt_echo@ba4c5dd9bb6aaa432abf4822607ce224bec51862
 * and @2a7b42a (reclaim redundant resident slots). The long-context lookup
 * follows the joint_ordered_long_context_512 variant.
 * One AIV per request group; MTP0/1/3 correspond to 1/2/4 query rows.
 * TopK=2048; capacity=4096/4096/8192. All group widths support up to 1 Mi tokens.
 *
 * RunOrderedPlan prepares each query group, assigns resident slots,
 * and publishes cache metadata. ReclaimRedundantSlots runs if the initial
 * miss assignment requires more slots.
 * Host integration and runtime tiling are registered separately.
 */

#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
#include "sparse_kv_plan_tiling.h"

#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(type)
#endif

namespace {

// Request buffers and unified-buffer layout.
using sparse_kv_plan_config::kPlanThreads;
using sparse_kv_plan_config::kBaselineBitmapTokens;
using sparse_kv_plan_config::kGroupControlWords;
using sparse_kv_plan_config::kLongContextLimit;
using sparse_kv_plan_config::kResidentBitmapLimit;
using sparse_kv_plan_config::kScanBankWords;
using sparse_kv_plan_config::kScanStorageWords;
using sparse_kv_plan_config::kTokensPerBitmapWord;
using sparse_kv_plan_config::kWarpSize;
constexpr uint32_t kWarpsPerBlock = kPlanThreads / kWarpSize;
static_assert(kWarpsPerBlock <= kWarpSize && (kWarpsPerBlock & (kWarpsPerBlock - 1U)) == 0U);
static_assert(kWarpsPerBlock <= kScanBankWords);

struct PlanBuffers {
    __gm__ int64_t *reqIds, *lastReqIds;
    __gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq;
    __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;
    __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;
    uint32_t topk, capacity;
    int32_t maxToken, blockSize;
    int64_t maxRequests;
};
enum RowMode : int32_t {
    kSingleRow = 0,
    kSharedLeader = 1,
    kSharedFollower = 2,
};
__simt_callee__ __aicore__ inline uint32_t ResidentPageCount(const PlanBuffers &buffers)
{
    const uint32_t blockSize = static_cast<uint32_t>(buffers.blockSize);
    if ((blockSize & (blockSize - 1U)) != 0U) {
        return buffers.capacity / blockSize;
    }
    uint32_t shift = 0U;
    for (uint32_t remaining = blockSize; remaining > 1U; remaining >>= 1U) {
        ++shift;
    }
    return buffers.capacity >> shift;
}

template <typename P>
__simt_callee__ __aicore__ inline uint32_t LowerBoundToken(P p, uint32_t n, int32_t token)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1U);
        if (p[mid] < token) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}
// Word offsets in the UB control region shared by Scalar, SIMT, and SIMD.
enum GroupWord : uint32_t {
    kGroupStatus = 0,
    kUniqueCount = 1,
    kRecordsPerSource = 3,
    kMergeRecordCount = 4,
    kNeedsQueryPacking = 6,
    kActiveSourceCount = 28,
    kMergeOutputOffset = 30,
    kEvictableSlots = 43,
    kHitSlots = 44,
    kMissingTokens = 45,
    kDescriptorTailCount = 46,
    kDescriptorCount = 47,
    kValidBegin = 48,
    kValidEnd = 56,
    kSourceRow = 64,
    kSourceLength = 72,
};

constexpr int32_t kGroupRejected = 0;
constexpr int32_t kGroupReady = 1;
constexpr int32_t kGroupCommitted = 2;
constexpr uint32_t kInvalidUid = 65535U;
constexpr uint32_t kLruFlagsWordOffset = 12288U;
constexpr uint32_t kReclaimedSlotsWordOffset = 16384U;
constexpr uint32_t kSlotOwnerMask = UINT16_MAX;
constexpr uint32_t kOwnerRankShift = 16U;
constexpr uint32_t kPackedSlotIndexMask = 0x3fffU;
constexpr uint32_t kResidentHitFlag = 0x4000U;
constexpr uint32_t kResidentInvalidFlag = 0x8000U;
constexpr uint32_t kPartitionInvalidFlag = 0x4000U;
constexpr uint32_t kMissOwnerFlag = 0x40000000U;

constexpr int32_t kNeedsReclaim = -2;
constexpr uint32_t kDefaultBitmapWords = kBaselineBitmapTokens / kTokensPerBitmapWord;
constexpr uint32_t kResidentDirectoryBins = 1024U;
static_assert(2U * kDefaultBitmapWords + 4096U / 2U <= sparse_kv_plan_config::kTwoWayMergeWorkspaceWords);
static_assert(2U * kDefaultBitmapWords + 8192U / 2U <= sparse_kv_plan_config::kFourWayMergeWorkspaceWords);
static_assert(kResidentDirectoryBins + 1U <= 2U * kDefaultBitmapWords);
// For the largest cache, the partition ends where the two reclaim buffers begin.
// The reclaim buffers are disjoint from the slot lookup that remains live.
static_assert(8192U + 8192U / 2U <= kLruFlagsWordOffset);
static_assert(kLruFlagsWordOffset + 8192U / 2U <= kReclaimedSlotsWordOffset);
static_assert(kReclaimedSlotsWordOffset + 8192U / 2U <= 2U * kDefaultBitmapWords);

__simt_callee__ __aicore__ inline uint32_t BitmapWordsForTokens(int32_t maxToken)
{
    return (static_cast<uint32_t>(maxToken) + kTokensPerBitmapWord - 1U) / kTokensPerBitmapWord;
}

__simt_callee__ __aicore__ inline uint32_t ResidentBitmapWordStride(int32_t maxToken)
{
    if (maxToken > kBaselineBitmapTokens && maxToken <= kResidentBitmapLimit) {
        return BitmapWordsForTokens(maxToken);
    }
    // Keep the existing placement for short bitmaps and directory lookup.
    return kDefaultBitmapWords;
}

__simt_callee__ __aicore__ inline __ubuf__ uint16_t *ResidentSlotLookup(__ubuf__ int32_t *workspace, int32_t maxToken)
{
    return reinterpret_cast<__ubuf__ uint16_t *>(workspace + 2U * ResidentBitmapWordStride(maxToken));
}

// Cold initialization.

// A reset starts with empty cache slots and assigns each token its compact index.
__simt_callee__ __aicore__ inline void InitializeColdGroup(const PlanBuffers &buffers, int64_t first, int64_t end,
                                                           __ubuf__ uint16_t *queryToUid,
                                                           __ubuf__ int32_t *uniqueTokens, __ubuf__ int32_t *lruStaging,
                                                           __ubuf__ int32_t *groupControl, uint32_t thread)
{
    constexpr uint32_t threadCount = kPlanThreads;
    const uint32_t uniqueCount = static_cast<uint32_t>(groupControl[kUniqueCount]),
                   rows = static_cast<uint32_t>(end - first);
    if (uniqueCount > buffers.capacity) {
        if (thread == 0U) {
            groupControl[kGroupStatus] = kGroupRejected;
        }
        asc_syncthreads();
        return;
    }
    const int64_t queryBase = first * static_cast<int64_t>(buffers.topk),
                  slotBase = first * static_cast<int64_t>(buffers.capacity);
    const uint32_t unused = buffers.capacity - uniqueCount, queryCount = rows * buffers.topk;
    // Reset has no valid resident/hit. Assign uid directly to identity slot uid.
    // This fuses cache initialization and fill; no intermediate owner/flags/partition.
    for (uint32_t slot = thread; slot < buffers.capacity; slot += threadCount) {
        const int32_t token = slot < uniqueCount ? uniqueTokens[slot] : -1;
        buffers.slotToToken[slotBase + slot] = token;
        lruStaging[slot] = static_cast<int32_t>(slot < unused ? uniqueCount + slot : slot - unused);
        if (slot < uniqueCount) {
            buffers.missTokens[queryBase + slot] = token;
            buffers.missSlots[queryBase + slot] = static_cast<int32_t>(slot);
        }
    }
    for (uint32_t i = thread; i < queryCount; i += threadCount) {
        const uint32_t uid = queryToUid[i];
        buffers.currentSlots[queryBase + i] = uid != kInvalidUid ? static_cast<int32_t>(uid) : -1;
    }
    // Stage the unused descriptor tail for the bulk writeback.
    asc_syncthreads(); // finish every unique token consumer before reusing unique as -1 buffer
    for (uint32_t i = thread; i < queryCount - uniqueCount; i += threadCount) {
        uniqueTokens[i] = -1;
    }
    const int32_t requestIndex = buffers.tokenToReq[first];
    const uint32_t pages = ResidentPageCount(buffers);
    for (uint32_t rowInGroup = 0; rowInGroup < rows; ++rowInGroup) {
        const int64_t row = first + rowInGroup;
        for (uint32_t page = thread; page < pages; page += threadCount) {
            buffers.residentBlockTable[row * pages + page] = static_cast<int32_t>(first * pages + page);
        }
        if (thread == 0U) {
            buffers.residentRows[row] = static_cast<int32_t>(first);
            const int64_t left = static_cast<int64_t>(uniqueCount) - static_cast<int64_t>(rowInGroup) * buffers.topk;
            buffers.missCount[row] = left <= 0           ? 0 :
                                     left < buffers.topk ? static_cast<int32_t>(left) :
                                                           static_cast<int32_t>(buffers.topk);
        }
        // Preserve the empirical thread-1 mode-publication workaround. No extra output.
        if (thread == 1U) {
            buffers.rowModes[row] = rows == 1U ? kSingleRow : (rowInGroup == 0U ? kSharedLeader : kSharedFollower);
        }
    }
    if (thread == 0U) {
        buffers.lastReqIds[first] = buffers.reqIds[requestIndex];
        groupControl[kDescriptorTailCount] = queryCount - uniqueCount;
        groupControl[kDescriptorCount] = uniqueCount;
        groupControl[kGroupStatus] = kGroupCommitted;
    }
    asc_syncthreads(); // finish all writes before this AIV reuses scratch for another group
}
// SPDX-License-Identifier: Apache-2.0

// Find the visible token ranges and initialize the query-to-UID map.
__simt_vf__ __aicore__ LAUNCH_BOUND(kPlanThreads) inline void PrepareQueryGroupVf(
    __ubuf__ int32_t *groupControl, __ubuf__ int32_t *groupData, const __gm__ int32_t *topkIndices,
    const __gm__ int32_t *visibleSeqLens, const __gm__ int32_t *tokenToReq, int64_t first, int64_t end, uint32_t topk,
    uint32_t capacity, int32_t maxToken, int64_t maxRequests)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t rows = end - first;
    const uint32_t queryCount = rows * topk;
    const int64_t queryBase = first * static_cast<int64_t>(topk);
    auto *uniqueTokens = groupData + capacity / 2U;
    auto *queryToUid = reinterpret_cast<__ubuf__ uint16_t *>(groupData);
    if (thread == 0U) {
        groupControl[kGroupStatus] = kGroupReady;
        groupControl[kUniqueCount] = 0;
        groupControl[kMergeRecordCount] = 0;
        groupControl[kNeedsQueryPacking] = 0;
        groupControl[kActiveSourceCount] = 0;
        // Split the merge workspace evenly between input and output records.
        groupControl[kMergeOutputOffset] = (rows == 4U ? sparse_kv_plan_config::kFourWayMergeWorkspaceWords :
                                                         sparse_kv_plan_config::kTwoWayMergeWorkspaceWords) /
                                           2U;
        const int32_t requestIndex = tokenToReq[first];
        // Merge records store the token and original query position in separate words.
        if (queryCount > capacity || queryCount > kInvalidUid || maxToken <= 0 || maxToken > kLongContextLimit ||
            requestIndex < 0 || requestIndex >= maxRequests) {
            groupControl[kGroupStatus] = kGroupRejected;
        }
        // The launch grid addresses fixed-width, contiguous request groups.
        for (int64_t row = first + 1; row < end; ++row) {
            if (tokenToReq[row] != requestIndex) {
                groupControl[kGroupStatus] = kGroupRejected;
            }
        }
        for (uint32_t r = 0; r < 8U; ++r) {
            groupControl[kValidBegin + r] = 0;
            groupControl[kValidEnd + r] = 0;
        }
        if (groupControl[kGroupStatus]) {
            for (uint32_t r = 0; r < rows; ++r) {
                const int64_t rowQueryBase = (first + r) * static_cast<int64_t>(topk);
                int32_t upper = visibleSeqLens[first + r];
                if (upper < 0) {
                    upper = 0;
                }
                if (upper > maxToken) {
                    upper = maxToken;
                }
                if (topkIndices[rowQueryBase + topk - 1U] < 0) {
                    // Find the first trailing padding entry.
                    uint32_t lo = 0, hi = topk;
                    while (lo < hi) {
                        const uint32_t mid = lo + (hi - lo) / 2U;
                        if (topkIndices[rowQueryBase + mid] >= 0) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    groupControl[kValidEnd + r] =
                        static_cast<int32_t>(LowerBoundToken(topkIndices + rowQueryBase, lo, upper));
                } else if (topk > 0U && topkIndices[rowQueryBase] >= 0 &&
                           topkIndices[rowQueryBase + topk - 1U] < upper) {
                    groupControl[kValidBegin + r] = 0;
                    groupControl[kValidEnd + r] = static_cast<int32_t>(topk);
                } else {
                    groupControl[kValidBegin + r] =
                        static_cast<int32_t>(LowerBoundToken(topkIndices + rowQueryBase, topk, 0));
                    groupControl[kValidEnd + r] =
                        static_cast<int32_t>(LowerBoundToken(topkIndices + rowQueryBase, topk, upper));
                }
                if (groupControl[kValidBegin + r] < groupControl[kValidEnd + r]) {
                    groupControl[kNeedsQueryPacking] = 1;
                }
            }
        }
        if (groupControl[kGroupStatus] && rows == 1U) {
            groupControl[kUniqueCount] = groupControl[kValidEnd] - groupControl[kValidBegin];
            groupControl[kNeedsQueryPacking] = 0;
        }
    }
    asc_syncthreads();
    // Full multi-row packets overwrite every query map entry below.
    const bool full = rows > 1U && groupControl[kGroupStatus] && groupControl[kValidBegin] == 0 &&
                      groupControl[kValidEnd] == topk && groupControl[kValidBegin + 1] == 0 &&
                      groupControl[kValidEnd + 1] == topk &&
                      (rows == 2U || (groupControl[kValidBegin + 2] == 0 && groupControl[kValidEnd + 2] == topk &&
                                      groupControl[kValidBegin + 3] == 0 && groupControl[kValidEnd + 3] == topk));
    if (full) {
        return;
    }
    if (groupControl[kGroupStatus]) {
        for (uint32_t queryIndex = thread; queryIndex < queryCount; queryIndex += kPlanThreads) {
            if (rows == 1U) {
                const uint32_t validBegin = static_cast<uint32_t>(groupControl[kValidBegin]);
                const bool valid =
                    queryIndex >= validBegin && queryIndex < static_cast<uint32_t>(groupControl[kValidEnd]);
                const int32_t token = topkIndices[queryBase + queryIndex];
                // Check only the valid interval so repeated padding stays legal.
                if (valid && queryIndex > validBegin && token <= topkIndices[queryBase + queryIndex - 1U]) {
                    asc_atomic_min(groupControl + kGroupStatus, kGroupRejected);
                }
                queryToUid[queryIndex] = valid ? static_cast<uint16_t>(queryIndex - validBegin) : uint16_t{65535};
                if (valid) {
                    uniqueTokens[queryIndex - validBegin] = token;
                }
            } else {
                queryToUid[queryIndex] = kInvalidUid;
            }
        }
    }
    asc_syncthreads();
}

// Pack each visible token with its original query position for hardware merge.
__simt_vf__ __aicore__ LAUNCH_BOUND(kPlanThreads) inline void PackMergeInputsVf(__ubuf__ int32_t *groupControl,
                                                                                __ubuf__ int32_t *packet,
                                                                                const __gm__ int32_t *topkIndices,
                                                                                int64_t first, int64_t end,
                                                                                uint32_t topk)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t rows = end - first;
    const int64_t queryBase = first * static_cast<int64_t>(topk);
    if (thread == 0U) {
        uint32_t activeSourceCount = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            if (groupControl[kValidBegin + r] < groupControl[kValidEnd + r]) {
                groupControl[kSourceRow + activeSourceCount++] = r;
            }
        }
        // Each active source has an aligned packet large enough for all its valid tokens.
        const uint32_t recordsPerSource =
            (static_cast<uint32_t>(groupControl[kMergeOutputOffset]) / (2U * activeSourceCount)) & ~3U;
        groupControl[kRecordsPerSource] = recordsPerSource;
        groupControl[kActiveSourceCount] = activeSourceCount;
        uint32_t total = 0;
        for (uint32_t k = 0; k < activeSourceCount; ++k) {
            uint32_t r = groupControl[kSourceRow + k], n = groupControl[kValidEnd + r] - groupControl[kValidBegin + r];
            groupControl[kSourceLength + k] = n;
            total += n;
        }
        groupControl[kMergeRecordCount] = total;
    }
    asc_syncthreads();
    const uint32_t recordsPerSource = groupControl[kRecordsPerSource],
                   activeSourceCount = groupControl[kActiveSourceCount];
    auto *score = reinterpret_cast<__ubuf__ float *>(packet);
    for (uint32_t i = thread; i < activeSourceCount * recordsPerSource; i += kPlanThreads) {
        uint32_t k = i / recordsPerSource, j = i % recordsPerSource;
        if (j < static_cast<uint32_t>(groupControl[kSourceLength + k])) {
            uint32_t r = groupControl[kSourceRow + k], pos = static_cast<uint32_t>(groupControl[kValidBegin + r]) + j;
            score[2U * i] = -static_cast<float>(topkIndices[queryBase + r * topk + pos]);
            packet[2U * i + 1U] = r * topk + pos;
        }
    }
    asc_syncthreads();
}

// SIMD deduplication and query mapping.

__simd_vf__ inline void DeduplicateMergePacket(__ubuf__ int32_t *records, __ubuf__ int32_t *data,
                                               __ubuf__ int32_t *groupControl, uint32_t recordCount,
                                               uint32_t queryCount, uint32_t capacity, uint32_t mapWords)
{
    using namespace AscendC;
    Reg::RegTensor<uint32_t> lane, zero, one, scoreIndices, queryIndices, previousIndex, lastIndex;
    Reg::RegTensor<uint32_t> flags, prefix, uniqueCount, uid, queryIndex, index, safeIndex, writeFlag, status;
    Reg::RegTensor<float> score, previous, lastScore, shiftScore, positive;
    Reg::RegTensor<int32_t> token;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg first, newToken, valid, fits, writeHead, chosen;
    // Unsqueeze(ALL) is exactly [0,1,...,63] for b32: no lane table in UB.
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(uniqueCount, uint32_t{0});
    Reg::Duplicate(lastScore, 1.0f);
    Reg::Compares<uint32_t, CMPMODE::EQ>(first, lane, uint32_t{0}, full);
    Reg::Adds(previousIndex, lane, uint32_t(-1), full);
    Reg::Select(previousIndex, zero, previousIndex, first); // no out-of-range register Gather
    static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::CAST_TRUNC};
    for (uint32_t base = 0; base < recordCount; base += 64U) {
        uint32_t remaining = recordCount - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);
        const uint32_t count = recordCount - base < 64U ? recordCount - base : 64U;
        Reg::Adds(scoreIndices, lane, base, full);
        Reg::ShiftLefts(scoreIndices, scoreIndices, int16_t{1}, full);
        Reg::Adds(queryIndices, scoreIndices, uint32_t{1}, full);
        Reg::Duplicate(score, 0.0f);
        Reg::Gather<float, float, uint32_t>(score, reinterpret_cast<__ubuf__ float *>(records), scoreIndices, valid);
        Reg::Gather<float, uint32_t>(shiftScore, score, previousIndex);
        Reg::Select(previous, lastScore, shiftScore, first);
        Reg::Compare<float, CMPMODE::NE>(newToken, score, previous, valid);
        Reg::Select(flags, one, zero, newToken);
        Reg::Select(flags, flags, zero, valid);
        // Rebuild all mask bits: no assumption about Compare's inactive lanes.
        Reg::Compares<uint32_t, CMPMODE::NE>(newToken, flags, uint32_t{0}, full);
        Reg::Unsqueeze<uint32_t>(prefix, newToken); // exclusive head count; last head not included here
        Reg::Add(prefix, prefix, flags, full);      // include every head, including lane63
        Reg::Add(prefix, prefix, uniqueCount, full);
        Reg::Adds(uid, prefix, uint32_t(-1), full);
        Reg::Neg(positive, score, full);
        Reg::Cast<int32_t, float, toInt>(token, positive, valid); // exact integers in [0,2^24)
        Reg::Compares<uint32_t, CMPMODE::LT>(fits, uid, capacity, full);
        Reg::Select(writeFlag, flags, zero, fits);
        Reg::Compares<uint32_t, CMPMODE::NE>(writeHead, writeFlag, uint32_t{0}, full);
        Reg::Adds(index, uid, mapWords, full);
        Reg::Adds(safeIndex, lane, mapWords + capacity, full);
        Reg::Select(index, index, safeIndex, writeHead); // even masked indices are unique
        Reg::Scatter<int32_t, uint32_t>(data, token, index, writeHead);
        Reg::Duplicate(queryIndex, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(queryIndex, reinterpret_cast<__ubuf__ uint32_t *>(records),
                                                  queryIndices, valid);
        Reg::Adds(safeIndex, lane, queryCount, full);
        Reg::Select(queryIndex, queryIndex, safeIndex, valid);
        // B32 mask selects even B16 lanes. Each low half stores one UID;
        // inactive odd-lane indices are distinct and inside allocated scratch.
        Reg::Adds(safeIndex, lane, queryCount + 64U, full);
        Reg::ShiftLefts(index, safeIndex, int16_t{16}, full);
        Reg::Or(index, index, queryIndex, full);
        Reg::Scatter<uint16_t, uint16_t>(reinterpret_cast<__ubuf__ uint16_t *>(data),
                                         reinterpret_cast<Reg::RegTensor<uint16_t> &>(uid),
                                         reinterpret_cast<Reg::RegTensor<uint16_t> &>(index), valid);
        Reg::Duplicate(lastIndex, uint32_t{63});
        Reg::Gather<uint32_t, uint32_t>(uniqueCount, prefix, lastIndex);
        Reg::Duplicate(lastIndex, count - 1U);
        Reg::Gather<float, uint32_t>(lastScore, score, lastIndex);
    }
    // Publish the deduplicated token count and capacity check in the control region.
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kUniqueCount}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(groupControl), uniqueCount, lane, chosen);
    Reg::Compares<uint32_t, CMPMODE::GE>(fits, uniqueCount, capacity + 1U, full);
    Reg::Select(status, zero, one, fits);
    Reg::Compares<uint32_t, CMPMODE::EQ>(chosen, lane, uint32_t{kGroupStatus}, full);
    Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(groupControl), status, lane, chosen);
}

// Both indexes occupy the completed merge packet. Each publishes the same
// packed result per physical slot before the assignment table reuses the index.
__simt_callee__ __aicore__ inline void LookupResidentsByBitmap(
    const __gm__ int32_t *residentTokens, const __ubuf__ int32_t *uniqueTokens, __ubuf__ int32_t *workspace,
    __ubuf__ uint16_t *slotLookup, uint32_t capacity, uint32_t uniqueCount, int32_t maxToken, int32_t stableEnd)
{
    const uint32_t thread = threadIdx.x;
    auto *masks = reinterpret_cast<__ubuf__ uint32_t *>(workspace);
    auto *firstUid = reinterpret_cast<__ubuf__ uint32_t *>(workspace + ResidentBitmapWordStride(maxToken));
    const uint32_t words = BitmapWordsForTokens(maxToken);
    for (uint32_t word = thread; word < words; word += kPlanThreads) {
        masks[word] = 0U;
    }
    asc_syncthreads();
    for (uint32_t uid = thread; uid < uniqueCount; uid += kPlanThreads) {
        const uint32_t token = static_cast<uint32_t>(uniqueTokens[uid]);
        const uint32_t word = token / kTokensPerBitmapWord;
        if (uid == 0U || static_cast<uint32_t>(uniqueTokens[uid - 1U]) / kTokensPerBitmapWord != word) {
            uint32_t mask = 0U;
            // The first token in each bitmap word owns both writes.
            for (uint32_t i = uid;
                 i < uniqueCount && static_cast<uint32_t>(uniqueTokens[i]) / kTokensPerBitmapWord == word; ++i) {
                mask |= uint32_t{1} << (static_cast<uint32_t>(uniqueTokens[i]) % kTokensPerBitmapWord);
            }
            firstUid[word] = uid;
            masks[word] = mask;
        }
    }
    asc_syncthreads();
    for (uint32_t slot = thread; slot < capacity; slot += kPlanThreads) {
        const int32_t token = residentTokens[slot];
        uint32_t encoded = token >= stableEnd ? kResidentInvalidFlag : 0U;
        if (token >= 0 && token < stableEnd) {
            const uint32_t tokenIndex = static_cast<uint32_t>(token);
            const uint32_t word = tokenIndex / kTokensPerBitmapWord;
            const uint32_t bit = uint32_t{1} << (tokenIndex % kTokensPerBitmapWord);
            const uint32_t mask = masks[word];
            if (mask & bit) {
                encoded = kResidentHitFlag | (firstUid[word] + __popc(mask & (bit - 1U)));
            }
        }
        slotLookup[slot] = static_cast<uint16_t>(encoded);
    }
    asc_syncthreads(); // All index reads complete before the shared workspace is reused.
}

__simt_callee__ __aicore__ inline void LookupResidentsByDirectory(
    const __gm__ int32_t *residentTokens, const __ubuf__ int32_t *uniqueTokens, __ubuf__ int32_t *directory,
    __ubuf__ uint16_t *slotLookup, uint32_t capacity, uint32_t uniqueCount, int32_t maxToken, int32_t stableEnd)
{
    const uint32_t thread = threadIdx.x;
    uint32_t bucketShift = 0U;
    while ((kResidentDirectoryBins << bucketShift) < static_cast<uint32_t>(maxToken)) {
        ++bucketShift;
    }
    // Directory entries mark the first UID of each token range, plus an end sentinel.
    for (uint32_t bucket = thread; bucket <= kResidentDirectoryBins; bucket += kPlanThreads) {
        directory[bucket] = LowerBoundToken(uniqueTokens, uniqueCount, static_cast<int32_t>(bucket << bucketShift));
    }
    asc_syncthreads();
    for (uint32_t slot = thread; slot < capacity; slot += kPlanThreads) {
        const int32_t token = residentTokens[slot];
        uint32_t encoded = token >= stableEnd ? kResidentInvalidFlag : 0U;
        if (token >= 0 && token < stableEnd) {
            const uint32_t bucket = static_cast<uint32_t>(token) >> bucketShift;
            const uint32_t begin = directory[bucket];
            const uint32_t end = directory[bucket + 1U];
            const uint32_t uid = begin + LowerBoundToken(uniqueTokens + begin, end - begin, token);
            if (uid < end && uniqueTokens[uid] == token) {
                encoded = kResidentHitFlag | uid;
            }
        }
        slotLookup[slot] = static_cast<uint16_t>(encoded);
    }
    asc_syncthreads(); // All index reads complete before the shared workspace is reused.
}

// Classify resident slots and select the last matching slot in LRU order.
__simt_callee__ __aicore__ inline void ClassifyResidentSlots(const PlanBuffers &buffers, int64_t first, int64_t end,
                                                             const __ubuf__ int32_t *uniqueTokens,
                                                             __ubuf__ int32_t *slotAssignments,
                                                             __ubuf__ int32_t *groupControl)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t capacity = buffers.capacity;
    const uint32_t uniqueCount = groupControl[kUniqueCount];
    const int64_t slotBase = first * capacity;
    // This region is reused for reclaim scratch after miss assignment.
    auto *lruFlags = reinterpret_cast<__ubuf__ uint16_t *>(slotAssignments + kLruFlagsWordOffset);
    int32_t stableEnd = buffers.stablePrefixLens[first];
    for (int64_t rowInGroup = first + 1; rowInGroup < end; ++rowInGroup) {
        if (buffers.stablePrefixLens[rowInGroup] < stableEnd) {
            stableEnd = buffers.stablePrefixLens[rowInGroup];
        }
    }
    if (stableEnd < 0) {
        stableEnd = 0;
    }
    if (stableEnd > buffers.maxToken) {
        stableEnd = buffers.maxToken;
    }
    auto *slotLookup = ResidentSlotLookup(slotAssignments, buffers.maxToken);
    const auto *residentTokens = buffers.slotToToken + slotBase;
    if (buffers.maxToken <= kResidentBitmapLimit) {
        LookupResidentsByBitmap(residentTokens, uniqueTokens, slotAssignments, slotLookup, capacity, uniqueCount,
                                buffers.maxToken, stableEnd);
    } else {
        LookupResidentsByDirectory(residentTokens, uniqueTokens, slotAssignments, slotLookup, capacity, uniqueCount,
                                   buffers.maxToken, stableEnd);
    }
    for (uint32_t i = thread; i < uniqueCount; i += kPlanThreads) {
        slotAssignments[i] = -1;
    }
    asc_syncthreads();
    for (uint32_t tile = 0; tile < capacity; tile += kPlanThreads) {
        const uint32_t order = tile + thread;
        int32_t slot = order < capacity ? buffers.lruSlots[slotBase + order] : -1;
        const bool valid = slot >= 0 && static_cast<uint32_t>(slot) < capacity;

        if (order < capacity && !valid) {
            asc_atomic_min(groupControl + kGroupStatus, kGroupRejected);
        }
        bool hit = false, invalid = false;
        if (valid) {
            const uint32_t encoded = slotLookup[slot];
            invalid = (encoded & kResidentInvalidFlag) != 0;
            hit = (encoded & kResidentHitFlag) != 0;
            if (hit) {
                asc_atomic_max(slotAssignments + (encoded & kPackedSlotIndexMask),
                               static_cast<int32_t>((order << kOwnerRankShift) | slot));
            }
        }
        if (order < capacity) {
            lruFlags[order] = valid ? static_cast<uint16_t>(slot | (hit ? kResidentHitFlag : 0U) |
                                                            (invalid ? kResidentInvalidFlag : 0U)) :
                                      uint16_t{65535};
        }
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kPlanThreads) inline void PrepareResidentPlanVf(PlanBuffers buffers, int64_t first,
                                                                                    int64_t end,
                                                                                    __ubuf__ int32_t *groupControl,
                                                                                    __ubuf__ int32_t *groupData)
{
    auto *uniqueTokens = groupData + buffers.capacity / 2U;
    auto *assignmentsOrLru = uniqueTokens + buffers.capacity;
    const int32_t requestIndex = buffers.tokenToReq[first];
    const int64_t requestId = buffers.reqIds[requestIndex];
    const bool reset = buffers.rowModes[first] != (end - first == 1 ? kSingleRow : kSharedLeader) ||
                       buffers.lastReqIds[first] != requestId;
    if (reset) {
        asc_syncthreads();
        auto *queryToUid = reinterpret_cast<__ubuf__ uint16_t *>(groupData);
        InitializeColdGroup(buffers, first, end, queryToUid, uniqueTokens, assignmentsOrLru, groupControl, threadIdx.x);
        return;
    }
    ClassifyResidentSlots(buffers, first, end, uniqueTokens, assignmentsOrLru, groupControl);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kPlanThreads) inline void CommitHotPlanVf(PlanBuffers buffers, int64_t first,
                                                                              int64_t end,
                                                                              __ubuf__ int32_t *groupControl,
                                                                              __ubuf__ int32_t *groupData)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t capacity = buffers.capacity;
    const uint32_t topk = buffers.topk;
    const uint32_t rows = end - first;
    const uint32_t queryCount = rows * topk;
    const uint32_t uniqueCount = groupControl[kUniqueCount];
    const uint32_t evicts = groupControl[kEvictableSlots];
    const uint32_t misses = groupControl[kMissingTokens];
    const int64_t queryBase = first * topk, slotBase = first * capacity;
    const int32_t requestIndex = buffers.tokenToReq[first];
    const int64_t requestId = buffers.reqIds[requestIndex];
    auto *uniqueTokens = groupData + capacity / 2U;
    auto *slotAssignments = uniqueTokens + capacity;
    const auto *partition = reinterpret_cast<__ubuf__ uint16_t *>(slotAssignments + capacity);
    // The complete assignment is available before any cache metadata is written.
    for (uint32_t uid = thread; uid < uniqueCount; uid += kPlanThreads) {
        const bool missing = (static_cast<uint32_t>(slotAssignments[uid]) & kMissOwnerFlag) != 0;
        const uint32_t rank = (static_cast<uint32_t>(slotAssignments[uid]) >> kOwnerRankShift) & kPackedSlotIndexMask;
        if (missing) {
            int32_t slot = slotAssignments[uid] & kSlotOwnerMask;
            buffers.slotToToken[slotBase + slot] = uniqueTokens[uid];
            buffers.missTokens[queryBase + rank] = uniqueTokens[uid];
            buffers.missSlots[queryBase + rank] = slot;
        }
    }
    // Read two query UIDs with one packed UB load.
    for (uint32_t pair = thread; pair < queryCount / 2; pair += kPlanThreads) {
        const uint32_t packed = reinterpret_cast<__ubuf__ uint32_t *>(groupData)[pair];
        const uint32_t lo = packed & kInvalidUid, hi = packed >> 16;
        buffers.currentSlots[queryBase + 2 * pair] = lo == kInvalidUid ? -1 : slotAssignments[lo] & kSlotOwnerMask;
        buffers.currentSlots[queryBase + 2 * pair + 1] = hi == kInvalidUid ? -1 : slotAssignments[hi] & kSlotOwnerMask;
    }
    // All readers of owner/unique have finished before output staging overwrites them.
    asc_syncthreads();
    auto *lruStaging = slotAssignments;
    for (uint32_t i = thread; i < queryCount - misses; i += kPlanThreads) {
        uniqueTokens[i] = -1;
    }
    const uint32_t unused = evicts - misses;
    for (uint32_t order = thread; order < capacity; order += kPlanThreads) {
        const uint32_t index = order < unused ? misses + order :
                               order < evicts ? order - unused :
                                                capacity - 1 - (order - evicts);
        const uint32_t entry = partition[index], slot = entry & kPackedSlotIndexMask;

        if (index >= misses && (entry & kPartitionInvalidFlag)) {
            buffers.slotToToken[slotBase + slot] = -1;
        }
        lruStaging[order] = static_cast<int32_t>(slot);
    }
    const uint32_t pages = ResidentPageCount(buffers);
    for (uint32_t rowInGroup = 0; rowInGroup < rows; ++rowInGroup) {
        int64_t row = first + rowInGroup;
        for (uint32_t p = thread; p < pages; p += kPlanThreads) {
            buffers.residentBlockTable[row * pages + p] = first * pages + p;
        }
        if (thread == 0) {
            buffers.residentRows[row] = first;
            int64_t left = static_cast<int64_t>(misses) - rowInGroup * topk;
            buffers.missCount[row] = left <= 0 ? 0 : left < topk ? left : topk;
        }
        if (thread == 1) {
            buffers.rowModes[row] = rows == 1 ? kSingleRow : (rowInGroup == 0 ? kSharedLeader : kSharedFollower);
        }
    }
    if (thread == 0) {
        groupControl[kDescriptorTailCount] = queryCount - misses;
        groupControl[kDescriptorCount] = misses;
        buffers.lastReqIds[first] = requestId;
        groupControl[kGroupStatus] = kGroupCommitted;
    }
    asc_syncthreads();
}

// SIMD compaction and MTE3 writeback.

// Packed halfword stores preserve the LRU order on each side of the partition.
__simd_callee__ inline void WriteGroupWord(__ubuf__ int32_t *groupControl, AscendC::Reg::RegTensor<uint32_t> &value,
                                           AscendC::Reg::RegTensor<uint32_t> &lane, uint32_t at,
                                           AscendC::Reg::MaskReg &full)
{
    AscendC::Reg::MaskReg chosen;
    AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::EQ>(chosen, lane, at, full);
    AscendC::Reg::Scatter<uint32_t, uint32_t>(reinterpret_cast<__ubuf__ uint32_t *>(groupControl), value, lane, chosen);
}
__simd_vf__ inline void AssignMissSlots(__ubuf__ int32_t *data, __ubuf__ int32_t *groupControl, uint32_t capacity,
                                        uint32_t uniqueCount)
{
    using namespace AscendC;
    const uint32_t ownerOffset = capacity / 2U + capacity, partitionOffset = ownerOffset + capacity;
    const uint32_t flagsOffset = ownerOffset + kLruFlagsWordOffset;
    Reg::RegTensor<uint32_t> lane, zero, one, idx, oddBit, packed, high, slotFlags, mask, slot, invalid;
    Reg::RegTensor<uint32_t> hitFlags, evictFlags, hitPrefix, ep, hitCount, evictableCount, hitRank, evictionRank, last,
        entry, safe, tmp, missCount, rank, inc, owner, index16, status;
    Reg::RegTensor<int32_t> signedOwner;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>(), odd, valid, hit, evict, selected, fits;
    Reg::Unsqueeze<uint32_t>(lane, full);
    Reg::Duplicate(zero, uint32_t{0});
    Reg::Duplicate(one, uint32_t{1});
    Reg::Duplicate(hitCount, uint32_t{0});
    Reg::Duplicate(evictableCount, uint32_t{0});
    Reg::Duplicate(last, uint32_t{63});
    Reg::Duplicate(mask, uint32_t{1});
    Reg::And(oddBit, lane, mask, full);
    Reg::Compares<uint32_t, CMPMODE::NE>(odd, oddBit, uint32_t{0}, full);
    for (uint32_t base = 0; base < capacity; base += 64U) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_UNPACK_B16>(
            slotFlags, reinterpret_cast<__ubuf__ uint32_t *>(data + flagsOffset) + (base / 2U));
        Reg::Compares<uint32_t, CMPMODE::NE>(valid, slotFlags, uint32_t{kInvalidUid}, full);
        Reg::Duplicate(mask, uint32_t{kResidentHitFlag});
        Reg::And(tmp, slotFlags, mask, full);
        Reg::Compares<uint32_t, CMPMODE::NE>(hit, tmp, uint32_t{0}, valid);
        Reg::Select(hitFlags, one, zero, hit);
        Reg::Select(evictFlags, one, zero, valid);
        Reg::Sub(evictFlags, evictFlags, hitFlags, full);
        Reg::Compares<uint32_t, CMPMODE::NE>(evict, evictFlags, uint32_t{0}, full);
        Reg::Unsqueeze<uint32_t>(hitPrefix, hit);
        Reg::Add(hitRank, hitPrefix, hitCount, full);

        Reg::Adds(evictionRank, lane, base, full);
        Reg::Sub(evictionRank, evictionRank, hitRank, full);
        Reg::Duplicate(tmp, capacity - 1U);
        Reg::Sub(idx, tmp, hitRank, full);
        Reg::Select(idx, idx, evictionRank, hit);
        Reg::Duplicate(mask, uint32_t{kPackedSlotIndexMask});
        Reg::And(entry, slotFlags, mask, full);
        Reg::Duplicate(mask, uint32_t{kResidentInvalidFlag});
        Reg::And(invalid, slotFlags, mask, full);
        Reg::ShiftRights(invalid, invalid, int16_t{1}, full);
        Reg::Or(entry, entry, invalid, full);
        // Only low halfwords selected by the B32 mask; inactive high-half indices safe.
        Reg::Adds(idx, idx, 2U * capacity, full);
        Reg::Select(idx, idx, lane, valid);
        Reg::Adds(safe, lane, uint32_t{64}, full);
        Reg::ShiftLefts(index16, safe, int16_t{16}, full);
        Reg::Or(index16, index16, idx, full);
        Reg::Scatter<uint16_t, uint16_t>(reinterpret_cast<__ubuf__ uint16_t *>(data + ownerOffset),
                                         reinterpret_cast<Reg::RegTensor<uint16_t> &>(entry),
                                         reinterpret_cast<Reg::RegTensor<uint16_t> &>(index16), valid);
        Reg::Add(hitRank, hitRank, hitFlags, full);
        Reg::Gather<uint32_t, uint32_t>(hitCount, hitRank, last);
        Reg::Duplicate(tmp, base + 64U);
        Reg::Sub(evictableCount, tmp, hitCount, full);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
    Reg::Duplicate(missCount, uint32_t{0});
    for (uint32_t base = 0; base < uniqueCount; base += 64U) {
        uint32_t remaining = uniqueCount - base;
        valid = Reg::UpdateMask<uint32_t>(remaining);

        Reg::LoadAlign<int32_t>(signedOwner, data + ownerOffset + base);
        Reg::Compares<int32_t, CMPMODE::LT>(selected, signedOwner, int32_t{0}, valid);
        Reg::Select(inc, one, zero, selected);
        Reg::Unsqueeze<uint32_t>(rank, selected);
        Reg::Add(rank, rank, missCount, full);
        Reg::Compare<uint32_t, CMPMODE::LT>(fits, rank, evictableCount, selected);
        Reg::ShiftRights(idx, rank, int16_t{1}, full);
        Reg::Adds(idx, idx, partitionOffset, full);
        Reg::Duplicate(tmp, partitionOffset);
        Reg::Select(idx, idx, tmp, fits);
        Reg::Duplicate(packed, uint32_t{0});
        Reg::Gather<uint32_t, uint32_t, uint32_t>(packed, reinterpret_cast<__ubuf__ uint32_t *>(data), idx, fits);
        Reg::Duplicate(mask, uint32_t{1});
        Reg::And(tmp, rank, mask, full);
        Reg::Compares<uint32_t, CMPMODE::NE>(odd, tmp, uint32_t{0}, full);
        Reg::ShiftRights(high, packed, int16_t{16}, full);
        Reg::Select(slot, high, packed, odd);
        Reg::Duplicate(mask, uint32_t{kPackedSlotIndexMask});
        Reg::And(slot, slot, mask, full);
        Reg::ShiftLefts(owner, rank, int16_t{16}, full);
        Reg::Or(owner, owner, slot, full);
        Reg::Duplicate(mask, uint32_t{kMissOwnerFlag});
        Reg::Or(owner, owner, mask, full);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM_B32>(
            reinterpret_cast<__ubuf__ uint32_t *>(data) + ownerOffset + base, owner, fits);
        Reg::Add(tmp, rank, inc, full);
        Reg::Gather<uint32_t, uint32_t>(missCount, tmp, last);
    }
    Reg::Add(tmp, evictableCount, hitCount, full);
    Reg::Compares<uint32_t, CMPMODE::EQ>(valid, tmp, capacity, full);
    Reg::Adds(tmp, evictableCount, uint32_t{1}, full);
    Reg::Compare<uint32_t, CMPMODE::LT>(selected, missCount, tmp, valid);
    Reg::Duplicate(status, static_cast<uint32_t>(kNeedsReclaim));
    Reg::Select(status, status, zero, valid);
    Reg::Select(status, one, status, selected);
    WriteGroupWord(groupControl, evictableCount, lane, kEvictableSlots, full);
    WriteGroupWord(groupControl, hitCount, lane, kHitSlots, full);
    WriteGroupWord(groupControl, missCount, lane, kMissingTokens, full);
    WriteGroupWord(groupControl, status, lane, kGroupStatus, full);
}

// SPDX-License-Identifier: Apache-2.0

// MTE3 consumer of the LRU and descriptor tails staged in UB by SIMT.
__aicore__ inline void CopyStagedOutputs(AscendC::TPipe &pipe, AscendC::LocalTensor<int32_t> local,
                                         __ubuf__ int32_t *groupControl, int64_t first, uint32_t topk,
                                         uint32_t capacity, GM_ADDR lruSlots, GM_ADDR missTokens, GM_ADDR missSlots)
{
    const uint32_t descriptorTailCount = static_cast<uint32_t>(groupControl[kDescriptorTailCount]),
                   misses = static_cast<uint32_t>(groupControl[kDescriptorCount]);
    const uint32_t uniqueOffset = kScanStorageWords + kGroupControlWords + capacity / 2U;
    const uint32_t ownerOffset = uniqueOffset + capacity;
    AscendC::GlobalTensor<int32_t> lru, missTokenOutput, missSlotOutput;
    lru.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lruSlots) + first * capacity, capacity);
    const int64_t tail = first * static_cast<int64_t>(topk) + misses;
    missTokenOutput.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missTokens) + tail, descriptorTailCount);
    missSlotOutput.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missSlots) + tail, descriptorTailCount);
    // Explicit SIMT/V -> MTE3 visibility; V_S only publishes the control values.
    auto ready = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ready);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ready);
    pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(ready);
    AscendC::DataCopyExtParams lruCopy(1, capacity * 4U, 0, 0, 0);
    AscendC::DataCopyPad(lru, local[ownerOffset], lruCopy);
    if (descriptorTailCount) {
        AscendC::DataCopyExtParams tailCopy(1, descriptorTailCount * 4U, 0, 0, 0);
        AscendC::DataCopyPad(missTokenOutput, local[uniqueOffset], tailCopy);
        AscendC::DataCopyPad(missSlotOutput, local[uniqueOffset], tailCopy);
    }
    // Both sources stay intact until ALL output copies finish. Includes the group
    // tail and kernel tail; no next VF/group may reuse these UB regions early.
    auto done = pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(done);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(done);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(done);
}

// Reclaim redundant resident slots.

__simt_callee__ __aicore__ inline uint32_t ScanSelectedLanes(__ubuf__ int32_t *scan, bool selected, uint32_t thread,
                                                             uint32_t &scanBank, uint32_t &selectedCount)
{
    const uint32_t lane = thread % kWarpSize;
    const uint32_t warp = thread / kWarpSize;
    const uint32_t ballot = asc_ballot(selected ? 1 : 0);
    const uint32_t laneRank = __popc(ballot & ((uint32_t{1} << lane) - 1U));
    auto *warpCounts = scan + scanBank * kScanBankWords;
    scanBank ^= 1U;
    if (lane == kWarpSize - 1U) {
        warpCounts[warp] = __popc(ballot);
    }
    // Alternating banks keeps the next scan's writes separate from these reads.
    asc_syncthreads();
    const int32_t warpCount = lane < kWarpsPerBlock ? warpCounts[lane] : 0;
    int32_t inclusive = warpCount;
#pragma unroll
    for (uint32_t offset = 1; offset < kWarpsPerBlock; offset <<= 1U) {
        const int32_t prior = asc_shfl_up(inclusive, offset, kWarpSize);
        if (lane >= offset) {
            inclusive += prior;
        }
    }
    selectedCount = asc_shfl(inclusive, kWarpsPerBlock - 1U, kWarpSize);
    return static_cast<uint32_t>(asc_shfl(inclusive - warpCount, warp, kWarpSize)) + laneRank;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kPlanThreads) inline void ReclaimRedundantSlots(__ubuf__ int32_t *scan,
                                                                                    __ubuf__ int32_t *groupControl,
                                                                                    uint32_t capacity, int32_t maxToken)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t evictableCount = groupControl[kEvictableSlots];
    const uint32_t hitCount = groupControl[kHitSlots];
    const uint32_t missingCount = groupControl[kMissingTokens];
    const uint32_t shortage = missingCount - evictableCount;
    const uint32_t uniqueCount = groupControl[kUniqueCount];
    auto *uidToSlot = groupControl + kGroupControlWords + capacity / 2U + capacity;
    auto *partition = reinterpret_cast<__ubuf__ uint16_t *>(uidToSlot + capacity);
    // Classification flags are no longer needed. Keep the hit partition intact
    // until all lanes have read it, using these two disjoint scratch buffers.
    auto *keptHits = reinterpret_cast<__ubuf__ uint16_t *>(uidToSlot + kLruFlagsWordOffset);
    auto *reclaimedSlots = reinterpret_cast<__ubuf__ uint16_t *>(uidToSlot + kReclaimedSlotsWordOffset);
    const auto *slotToUid = ResidentSlotLookup(uidToSlot, maxToken);

    uint32_t scanBank = 0;
    uint32_t redundantCount = 0;
    for (uint32_t base = 0; base < hitCount; base += kPlanThreads) {
        const uint32_t hitRank = base + thread;
        uint16_t entry = 0;
        bool redundant = false;
        if (hitRank < hitCount) {
            entry = partition[capacity - 1U - hitRank];
            const uint32_t slot = entry & kPackedSlotIndexMask;
            const uint32_t uid = slotToUid[slot] & kPackedSlotIndexMask;
            // Classification chose the last matching slot in LRU order.
            redundant = (static_cast<uint32_t>(uidToSlot[uid]) & kSlotOwnerMask) != slot;
        }
        uint32_t tileCount = 0;
        const uint32_t reclaimRank = redundantCount + ScanSelectedLanes(scan, redundant, thread, scanBank, tileCount);
        if (hitRank < hitCount) {
            if (redundant && reclaimRank < shortage) {
                reclaimedSlots[reclaimRank] = entry;
            } else {
                const uint32_t reclaimedBefore = reclaimRank < shortage ? reclaimRank : shortage;
                keptHits[hitRank - reclaimedBefore] = entry;
            }
        }
        redundantCount += tileCount;
    }
    asc_syncthreads();
    if (redundantCount < shortage) {
        if (thread == 0U) {
            groupControl[kGroupStatus] = kGroupRejected;
        }
        return;
    }
    for (uint32_t i = thread; i < hitCount - shortage; i += kPlanThreads) {
        partition[capacity - 1U - i] = keptHits[i];
    }
    for (uint32_t i = thread; i < shortage; i += kPlanThreads) {
        partition[evictableCount + i] = reclaimedSlots[i];
    }
    asc_syncthreads();

    // Existing assignments use the old evictable slots. Extend that sequence
    // with the reclaimed slots, preserving missing-token and LRU order.
    uint32_t unassignedCount = 0;
    for (uint32_t base = 0; base < uniqueCount; base += kPlanThreads) {
        const uint32_t uid = base + thread;
        const bool unassigned = uid < uniqueCount && uidToSlot[uid] < 0;
        uint32_t tileCount = 0;
        const uint32_t rank = unassignedCount + ScanSelectedLanes(scan, unassigned, thread, scanBank, tileCount);
        if (unassigned && rank < shortage) {
            const uint32_t missRank = evictableCount + rank;
            const uint32_t slot = reclaimedSlots[rank] & kPackedSlotIndexMask;
            uidToSlot[uid] = static_cast<int32_t>(kMissOwnerFlag | (missRank << kOwnerRankShift) | slot);
        }
        unassignedCount += tileCount;
    }
    if (thread == 0U) {
        groupControl[kEvictableSlots] = missingCount;
        groupControl[kHitSlots] = static_cast<int32_t>(hitCount - shortage);
        groupControl[kGroupStatus] = unassignedCount == shortage ? kGroupReady : kGroupRejected;
    }
    // Publish the complete slot assignment before the hot path writes GM.
    asc_syncthreads();
}

// Request scheduling and hardware merge.

__aicore__ inline void WaitForVector(AscendC::TPipe &pipe)
{
    auto id = pipe.AllocEventID<AscendC::HardEvent::V_S>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(id);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::V_S>(id);
}
__aicore__ inline void PublishScalarToVector(AscendC::TPipe &pipe)
{
    auto id = pipe.AllocEventID<AscendC::HardEvent::S_V>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(id);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
    pipe.ReleaseEventID<AscendC::HardEvent::S_V>(id);
}

__aicore__ inline __ubuf__ int32_t *MergePackedQueries(AscendC::TPipe &pipe, AscendC::LocalTensor<float> mergeStorage,
                                                       __ubuf__ int32_t *records, uint32_t packetOffset,
                                                       const __ubuf__ int32_t *groupControl)
{
    const uint32_t activeSourceCount = static_cast<uint32_t>(groupControl[kActiveSourceCount]);
    const uint32_t recordsPerSource = static_cast<uint32_t>(groupControl[kRecordsPerSource]);
    if (activeSourceCount > 1U) {
        const uint32_t outputOffset = static_cast<uint32_t>(groupControl[kMergeOutputOffset]);
        AscendC::MrgSortSrcList<float> lists{
            mergeStorage[packetOffset], mergeStorage[packetOffset + 2U * recordsPerSource],
            mergeStorage[packetOffset + (activeSourceCount > 2U ? 4U * recordsPerSource : 0U)],
            mergeStorage[packetOffset + (activeSourceCount > 3U ? 6U * recordsPerSource : 0U)]};
        uint16_t sourceLengths[4] = {};
        uint32_t sourceConsumed[4] = {};
        for (uint32_t k = 0; k < activeSourceCount; ++k) {
            sourceLengths[k] = static_cast<uint16_t>(groupControl[kSourceLength + k]);
        }
        // All visible records fit the packet. Keep the established merge -> VF
        // event handoff, with no readback of consumptions or refill loop.
        AscendC::MrgSort<float, false>(mergeStorage[packetOffset + outputOffset], lists, sourceLengths, sourceConsumed,
                                       static_cast<uint16_t>((1U << activeSourceCount) - 1U), 1);
        WaitForVector(pipe);
        PublishScalarToVector(pipe);
        records += outputOffset;
    }
    return records;
}

__aicore__ inline void RunOrderedPlan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                      GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR activeRows,
                                      GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots, GM_ADDR currentSlots,
                                      GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                                      GM_ADDR compactWorkspace, GM_ADDR rowModes, GM_ADDR residentRows,
                                      GM_ADDR residentBlockTable, GM_ADDR tiling)
{
    SparseKvPlanTiling planTiling;
    const auto *gm = reinterpret_cast<const __gm__ SparseKvPlanTiling *>(tiling);
    planTiling.maxRows = gm->maxRows;
    planTiling.topk = gm->topk;
    planTiling.capacity = gm->capacity;
    planTiling.maxToken = gm->maxToken;
    planTiling.maxRequests = gm->maxRequests;
    planTiling.blockSize = gm->blockSize;
    planTiling.localMemoryBytes = gm->localMemoryBytes;
    const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t *>(activeRows);
    const int64_t rows = requestedRows <= 0                 ? 0 :
                         requestedRows < planTiling.maxRows ? requestedRows :
                                                              planTiling.maxRows;
    int64_t blockIndex = AscendC::GetBlockIdx(), blockCount = AscendC::GetBlockNum();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, planTiling.localMemoryBytes);
    auto local = storage.Get<int32_t>();
    auto *scan = reinterpret_cast<__ubuf__ int32_t *>(local.GetPhyAddr());
    auto *groupControl = scan + kScanStorageWords;
    // The Host accepts only 1, 2 or 4 allocated rows per request. A partial
    // active request group is an input error. An inactive trailing group is OK.
    const int64_t rowsPerRequest =
        planTiling.capacity == 8192U ? 4 : (planTiling.maxRows == planTiling.maxRequests ? 1 : 2);
    auto *requestIndices = reinterpret_cast<const __gm__ int32_t *>(tokenToReq);
    auto *errors = reinterpret_cast<__gm__ int32_t *>(compactWorkspace);
    if (requestedRows < 0 || requestedRows > planTiling.maxRows || (rows & (rowsPerRequest - 1)) != 0) {
        if (blockIndex == 0) {
            errors[0] = -1;
        }
        return;
    }
    const uint32_t topk = static_cast<uint32_t>(planTiling.topk);
    const uint32_t capacity = static_cast<uint32_t>(planTiling.capacity);
    const uint32_t queryCount = static_cast<uint32_t>(rowsPerRequest) * topk;
    const uint32_t mapWords = capacity / 2U;
    const PlanBuffers buffers{reinterpret_cast<__gm__ int64_t *>(reqIds),
                              reinterpret_cast<__gm__ int64_t *>(lastReqIds),
                              reinterpret_cast<__gm__ int32_t *>(topkIndices),
                              reinterpret_cast<__gm__ int32_t *>(stablePrefixLens),
                              reinterpret_cast<__gm__ int32_t *>(visibleSeqLens),
                              reinterpret_cast<__gm__ int32_t *>(tokenToReq),
                              reinterpret_cast<__gm__ int32_t *>(slotToToken),
                              reinterpret_cast<__gm__ int32_t *>(lruSlots),
                              reinterpret_cast<__gm__ int32_t *>(currentSlots),
                              reinterpret_cast<__gm__ int32_t *>(missCount),
                              reinterpret_cast<__gm__ int32_t *>(missTokens),
                              reinterpret_cast<__gm__ int32_t *>(missSlots),
                              reinterpret_cast<__gm__ int32_t *>(rowModes),
                              reinterpret_cast<__gm__ int32_t *>(residentRows),
                              reinterpret_cast<__gm__ int32_t *>(residentBlockTable),
                              topk,
                              capacity,
                              static_cast<int32_t>(planTiling.maxToken),
                              planTiling.blockSize,
                              planTiling.maxRequests};
    auto *groupData = groupControl + kGroupControlWords;
    const uint32_t packetOffset = kScanStorageWords + kGroupControlWords + mapWords + capacity;
    auto mergeStorage = storage.Get<float>();
    for (int64_t first = blockIndex * rowsPerRequest; first < rows; first += blockCount * rowsPerRequest) {
        const int64_t end = first + rowsPerRequest;
        const int32_t request = requestIndices[first];
        if ((first > 0 && requestIndices[first - 1] == request) || (end < rows && requestIndices[end] == request)) {
            errors[first] = -1;
            continue;
        }
        asc_vf_call<PrepareQueryGroupVf>(dim3(kPlanThreads), groupControl, groupData, buffers.topkIndices,
                                         buffers.visibleSeqLens, buffers.tokenToReq, first, end, topk, capacity,
                                         buffers.maxToken, buffers.maxRequests);
        WaitForVector(pipe);
        if (groupControl[kGroupStatus] == kGroupReady && groupControl[kNeedsQueryPacking]) {
            asc_vf_call<PackMergeInputsVf>(dim3(kPlanThreads), groupControl, scan + packetOffset, buffers.topkIndices,
                                           first, end, topk);
            WaitForVector(pipe);
            auto *records = MergePackedQueries(pipe, mergeStorage, scan + packetOffset, packetOffset, groupControl);
            asc_vf_call<DeduplicateMergePacket>(records, groupData, groupControl,
                                                static_cast<uint32_t>(groupControl[kMergeRecordCount]), queryCount,
                                                capacity, mapWords);
            WaitForVector(pipe);
        }
        if (groupControl[kGroupStatus] == kGroupReady) {
            asc_vf_call<PrepareResidentPlanVf>(dim3(kPlanThreads), buffers, first, end, groupControl, groupData);
            WaitForVector(pipe);
            if (groupControl[kGroupStatus] == kGroupReady) {
                asc_vf_call<AssignMissSlots>(groupData, groupControl, capacity,
                                             static_cast<uint32_t>(groupControl[kUniqueCount]));
                WaitForVector(pipe);
                if (groupControl[kGroupStatus] == kNeedsReclaim) {
                    asc_vf_call<ReclaimRedundantSlots>(dim3(kPlanThreads), scan, groupControl, capacity,
                                                       buffers.maxToken);
                    WaitForVector(pipe);
                }
                if (groupControl[kGroupStatus] == kGroupReady) {
                    asc_vf_call<CommitHotPlanVf>(dim3(kPlanThreads), buffers, first, end, groupControl, groupData);
                    WaitForVector(pipe);
                }
            }
        }
        if (groupControl[kGroupStatus] == kGroupCommitted) {
            CopyStagedOutputs(pipe, local, groupControl, first, topk, capacity, lruSlots, missTokens, missSlots);
        } else {
            errors[first] = -1;
        }
    }
}

} // namespace

extern "C" __global__ __aicore__ void sparse_kv_plan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                                     GM_ADDR visibleSeqLens, GM_ADDR tokenToReq, GM_ADDR activeRows,
                                                     GM_ADDR lastReqIds, GM_ADDR slotToToken, GM_ADDR lruSlots,
                                                     GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens,
                                                     GM_ADDR missSlots, GM_ADDR compactWorkspace, GM_ADDR rowModes,
                                                     GM_ADDR residentRows, GM_ADDR residentBlockTable,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SparseKvPlanTiling);
    AscendC::InitSocState();
    RunOrderedPlan(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, activeRows, lastReqIds,
                   slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace, rowModes,
                   residentRows, residentBlockTable, tiling);
}
