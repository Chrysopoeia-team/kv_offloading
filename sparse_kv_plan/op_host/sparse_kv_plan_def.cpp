// SPDX-License-Identifier: Apache-2.0

#include "register/op_def_registry.h"

#include <array>

namespace {

constexpr std::array<const char *, 6> kInputs{"req_ids",          "topk_indices", "stable_prefix_lens",
                                             "visible_seq_lens", "token_to_req", "active_rows"};
constexpr std::array<const char *, 11> kOutputs{
    "last_req_ids", "slot_to_token", "lru_slots", "current_slots", "miss_count", "miss_tokens",
    "miss_slots", "plan_workspace", "row_modes", "resident_rows", "resident_block_table"};

}  // namespace

namespace ops {

class SparseKvPlan : public OpDef
{
public:
    explicit SparseKvPlan(const char *name) : OpDef(name)
    {
        for (size_t i = 0; i < kInputs.size(); ++i) {
            this->Input(kInputs[i])
                .ParamType(REQUIRED)
                .DataType({i == 0 ? ge::DT_INT64 : ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (size_t i = 0; i < kOutputs.size(); ++i) {
            this->Output(kOutputs[i])
                .ParamType(REQUIRED)
                .DataType({i == 0 ? ge::DT_INT64 : ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Attr("capacity").AttrType(REQUIRED).Int();
        this->Attr("max_token").AttrType(REQUIRED).Int();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "sparse_kv_plan")
            .ExtendCfgInfo("opInterface.value", "sparse_kv_plan");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(SparseKvPlan);

}  // namespace ops
