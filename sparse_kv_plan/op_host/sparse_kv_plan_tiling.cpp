// SPDX-License-Identifier: Apache-2.0

#include "../op_kernel/sparse_kv_plan_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>

namespace {

bool MatchesShape(const gert::StorageShape *shape, const gert::Shape &expected)
{
    return shape && shape->GetOriginShape() == expected && shape->GetStorageShape() == expected;
}

ge::graphStatus PlanTiling(gert::TilingContext *context)
{
    const auto *requests = context->GetInputShape(0);
    const auto *topk = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *capacity = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *maxToken = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    const auto *blockSize = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    if (!requests || !topk || !capacity || !maxToken || !blockSize || *blockSize <= 0 || *blockSize > INT32_MAX ||
        !context->GetPlatformInfo() || !workspace || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(SparseKvPlanTiling)) {
        return ge::GRAPH_FAILED;
    }

    const auto &requestShape = requests->GetStorageShape();
    const auto &topkShape = topk->GetStorageShape();
    if (requestShape.GetDimNum() != 1 || topkShape.GetDimNum() != 2 || topkShape.GetDim(1) != 2048) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<SparseKvPlanTiling>();
    uint64_t workspaceWords = 0;
    if (!tiling ||
        !sparse_kv_plan_config::MakeJointTiling(requestShape.GetDim(0), topkShape.GetDim(0), *capacity, *maxToken,
                                                static_cast<int32_t>(*blockSize), *tiling, workspaceWords)) {
        return ge::GRAPH_FAILED;
    }
    const auto rows = topkShape.GetDim(0);
    const gert::Shape rowShape{rows};
    const gert::Shape cacheShape{rows, *capacity};
    const std::array<gert::Shape, 6> inputs{requestShape, topkShape, rowShape, rowShape, rowShape, gert::Shape{1}};
    const std::array<gert::Shape, 11> outputs{rowShape,
                                              cacheShape,
                                              cacheShape,
                                              topkShape,
                                              rowShape,
                                              topkShape,
                                              topkShape,
                                              gert::Shape{static_cast<int64_t>(workspaceWords)},
                                              rowShape,
                                              rowShape,
                                              gert::Shape{rows, *capacity / *blockSize}};
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!MatchesShape(context->GetInputShape(i), inputs[i])) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!MatchesShape(context->GetOutputShape(i), outputs[i])) {
            return ge::GRAPH_FAILED;
        }
    }
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const auto cores = std::min<int64_t>(64, platform.GetCoreNumAiv());
    if (cores <= 0 || context->SetDynUBufSize(tiling->localMemoryBytes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(static_cast<uint32_t>(std::min(tiling->maxRequests, cores)));
    context->SetTilingKey(0);
    workspace[0] = 0;
    raw->SetDataSize(sizeof(*tiling));
    return ge::GRAPH_SUCCESS;
}

struct PlanCompileInfo {};

ge::graphStatus ParsePlanTiling(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(SparseKvPlan).Tiling(PlanTiling).TilingParse<PlanCompileInfo>(ParsePlanTiling);
