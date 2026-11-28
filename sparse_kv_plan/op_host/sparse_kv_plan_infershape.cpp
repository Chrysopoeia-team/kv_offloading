// SPDX-License-Identifier: Apache-2.0

#include "register/op_impl_registry.h"

#include <array>
#include <limits>

namespace {

constexpr size_t kOutputCount = 11;

ge::graphStatus PlanInferShape(gert::InferShapeContext *context)
{
    const auto *topk = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *capacity = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *blockSize = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    if (!topk || topk->GetDimNum() != 2 || !capacity || !blockSize || *blockSize <= 0 ||
        (*capacity != 4096 && *capacity != 8192) || *capacity % *blockSize != 0) {
        return ge::GRAPH_FAILED;
    }
    const auto rows = topk->GetDim(0);
    if (rows > std::numeric_limits<int64_t>::max() / 1024) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape rowShape{rows};
    const gert::Shape cacheShape{rows, *capacity};
    const gert::Shape workspaceShape{rows < 0 ? -1 : rows * 1024};
    const gert::Shape blockShape{rows, *capacity / *blockSize};
    const std::array<gert::Shape, kOutputCount> shapes{rowShape, cacheShape,     cacheShape, *topk,    rowShape,  *topk,
                                                       *topk,    workspaceShape, rowShape,   rowShape, blockShape};
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto *output = context->GetOutputShape(i);
        if (!output) {
            return ge::GRAPH_FAILED;
        }
        *output = shapes[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PlanInferDataType(gert::InferDataTypeContext *context)
{
    for (size_t i = 0; i < kOutputCount; ++i) {
        context->SetOutputDataType(i, i == 0 ? ge::DT_INT64 : ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(SparseKvPlan).InferShape(PlanInferShape).InferDataType(PlanInferDataType);
