/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef SPARSE_KV_PLAN_TILING_H
#define SPARSE_KV_PLAN_TILING_H

#include <climits>
#include <cstdint>

namespace sparse_kv_plan_config {

constexpr uint32_t kPlanThreads = 512U;
constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kScanBankWords = 32U;
// Keep the established aligned offset of the group control and merge buffers.
constexpr uint32_t kScanStorageWords = 2U * kScanBankWords + 8U;
constexpr uint32_t kGroupControlWords = 128U;
constexpr uint32_t kTwoWayMergeWorkspaceWords = 24576U;
constexpr uint32_t kFourWayMergeWorkspaceWords = 32768U;
constexpr int32_t kBaselineBitmapTokens = 358400;
constexpr int32_t kResidentBitmapLimit = 512 * 1024;
constexpr int32_t kLongContextLimit = 1024 * 1024;
constexpr uint32_t kTokensPerBitmapWord = 32U;

static_assert(kPlanThreads % kWarpSize == 0U);
static_assert(kScanStorageWords * sizeof(int32_t) % 32U == 0U);

} // namespace sparse_kv_plan_config

// Shared byte layout for CANN runtime tiling and the device kernel.
struct SparseKvPlanTiling {
    int64_t maxRows;
    int64_t topk;
    int64_t capacity;
    int64_t maxToken;
    int64_t maxRequests;
    int32_t blockSize;
    uint32_t localMemoryBytes;
};
static_assert(sizeof(SparseKvPlanTiling) == 48, "Sparse KV Plan tiling layout");
namespace sparse_kv_plan_config {

inline bool MakeJointTiling(int64_t maxRequests, int64_t maxRows, int64_t capacity, int64_t maxToken, int32_t blockSize,
                            SparseKvPlanTiling &tiling, uint64_t &workspaceInt32)
{
    if (maxRequests <= 0 || maxRows <= 0 || maxRows % maxRequests != 0 || maxToken <= 0 ||
        maxToken > kLongContextLimit || blockSize <= 0) {
        return false;
    }
    const int64_t rowsPerRequest = maxRows / maxRequests;
    const bool fourRows = rowsPerRequest == 4;
    if ((rowsPerRequest != 1 && rowsPerRequest != 2 && !fourRows) || capacity != (fourRows ? 8192 : 4096) ||
        capacity % blockSize != 0 || maxRows > INT32_MAX / (capacity / blockSize)) {
        return false;
    }

    constexpr uint64_t topK = 2048;
    const uint64_t residentCapacity = static_cast<uint64_t>(capacity);
    uint64_t reusableWords = fourRows ? kFourWayMergeWorkspaceWords : kTwoWayMergeWorkspaceWords;
    if (maxToken > kBaselineBitmapTokens && maxToken <= kResidentBitmapLimit) {
        // Use the shared workspace for the bitmap, first UID per word, and
        // packed resident lookup. Grow only when that layout needs more space.
        const uint64_t bitmapWords = (maxToken + kTokensPerBitmapWord - 1U) / kTokensPerBitmapWord;
        const uint64_t lookupWords = 2U * bitmapWords + residentCapacity / 2U;
        if (lookupWords > reusableWords) {
            reusableWords = lookupWords;
        }
    }
    const uint64_t scratchWords =
        kScanStorageWords + kGroupControlWords + residentCapacity / 2U + residentCapacity + reusableWords;
    constexpr uint64_t ubAlignment = 32U;
    const uint32_t dynamicUb =
        static_cast<uint32_t>((scratchWords * sizeof(int32_t) + ubAlignment - 1U) & ~(ubAlignment - 1U));
    tiling = {maxRows, topK, capacity, maxToken, maxRequests, blockSize, dynamicUb};
    // The public workspace tensor retains its shape and per-row error flags.
    workspaceInt32 = static_cast<uint64_t>(maxRows) * topK / 2U;
    return true;
}

} // namespace sparse_kv_plan_config

#endif
