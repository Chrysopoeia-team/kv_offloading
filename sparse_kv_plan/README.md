# Sparse KV Plan

`SparseKvPlan` 在 Ascend950 上消费按逻辑 token 位置升序排列的 TopK，更新 KV 缓存的槽位及 LRU，并输出本轮需要搬运的描述符。调用方可使用保序 QLI V1 或其他遵守有序 TopK 输入约定的生产者。

本目录注册一个算子 `SparseKvPlan`。CANN 生成的 `aclnn_sparse_kv_plan.h` 提供两阶段调用 `aclnnSparseKvPlanGetWorkspaceSize` 和 `aclnnSparseKvPlan`。设备端采用 512 线程执行有序规划。常规路径复用命中槽。可淘汰槽不足时，保留每个命中 token 在 LRU 顺序中的最后一个槽，再按 LRU 顺序回收刚好够用的重复副本，补齐本组的 miss 分配。源码来自 [ops-transformer 的 83a8ae7e 提交](https://gitcode.com/lingyuehjh/ops-transformer/tree/83a8ae7ef33f41a29d114a00c7d1d7a909393b58/attention/sparse_kv_plan)。许可文本见 [CANN Open Software License Agreement](LICENSE)、[Apache-2.0](docs/Apache-2.0.txt) 和 [MulanPSL-2.0](docs/MulanPSL-2.0.txt)。

## 接入

设 `R` 为请求数，`Q` 为分配的 query 行数，`K=2048`，`C` 为每个 resident 缓存容量。`block_size` 为 `C` 的正因数。

| 每请求连续 query 行数 | `C` |
| --- | ---: |
| 1 或 2（MTP0／1） | 4096 |
| 4（MTP3） | 8192 |

`max_token` 是本次调用的 token 编号排他上界。MTP0／1／3 均支持 1～1048576。多行路径先对有序 TopK 归并去重。热缓存查找在 `max_token <= 524288` 时用位图，超过时用目录＋精确二分。冷启动直接分槽。

Ordered V1 在 TND、单 key head 配置下的 `sparse_indices` 形状为 `[Q,1,K]`，有效索引严格递增且尾部为 `-1`。接入时取 `indices.squeeze(1)` 的连续 `[Q,K]` 视图。调用方需保证 TopK、缓存 token 和可见范围使用同一套逻辑坐标。Plan 的 `resident_block_table` 映射设备缓存，独立于 Indexer 的源页表。

| 输入 | Shape | 说明 |
| --- | --- | --- |
| `req_ids` | `[R]` | 请求身份，INT64 |
| `topk_indices` | `[Q,K]` | 有序逻辑 token 位置 |
| `stable_prefix_lens`, `visible_seq_lens` | `[Q]` | 稳定前缀与因果可见范围，均为排他上界 |
| `token_to_req` | `[Q]` | 每行到 `req_ids` 的下标，同一组连续 |
| `active_rows` | `[1]` | 有效行是完整请求组组成的前缀 |

| 调用方预分配的输出或状态 | Shape | 首次调用与消费约定 |
| --- | --- | --- |
| `last_req_ids` | `[Q]` | INT64，初始化为 `-1` |
| `slot_to_token` | `[Q,C]` | 初始化为 `-1` |
| `lru_slots` | `[Q,C]` | 每行初始化为 `[0,C)` 升序 |
| `current_slots` | `[Q,K]` | 对应每个 TopK 位置的缓存槽，无效为 `-1` |
| `miss_count`, `miss_tokens`, `miss_slots` | `[Q]`, `[Q,K]`, `[Q,K]` | 按每行 `miss_count` 读取有效前缀，联合组的描述符可跨组内多行 |
| `plan_workspace` | `[Q*1024]` | 前 `Q` 项初始化为 `INT32_MAX`。任意负值表示拒绝输入，调用方应停止消费结果 |
| `row_modes` | `[Q]` | 初始化为 0，0/1/2 分别表示独立行、联合首行、联合跟随行 |
| `resident_rows`, `resident_block_table` | `[Q]`, `[Q,C/block_size]` | 下游据此定位实际缓存行和页 |

表中其余张量为 INT32。所有张量使用同设备、连续的 ND 存储，缓冲区互不重叠。每轮在同一 stream 上调用，后续轮次继续使用已更新状态。调用方另外按 ACLNN 返回的大小分配其运行时 workspace，即使返回大小为 0 仍需保留表中的 `plan_workspace` 张量。

调用方应保证上一轮已经完成对同一块缓存的读取，再让本轮按新计划覆写槽。跨 stream 执行时，通过事件建立上一轮读、下一轮搬运写和下一轮注意力读之间的顺序。

## 构建和验证

本目录的 CMake 使用 `ops-transformer` 的算子构建规则。在 Ascend950 环境加载 CANN 后，使用前述固定提交的 `ops-transformer` 工作树，从该工作树的根目录执行以下命令。设备检查通过 ACLNN 调用 Plan，测试直接生成有序 TopK。

```bash
bash build.sh --pkg --soc=ascend950 --ops=sparse_kv_plan --vendor_name=sparsekvplan
PLAN_INSTALL=$(mktemp -d)
bash build_out/cann-ops-transformer-sparsekvplan*.run --install-path="$PLAN_INSTALL"
source "$PLAN_INSTALL/vendors/sparsekvplan_transformer/bin/set_env.bash"
python3 attention/sparse_kv_plan/tests/check_sparse_kv_plan.py \
  --device 0 --opapi "$PLAN_INSTALL/vendors/sparsekvplan_transformer/op_api/lib/libcust_opapi.so"
```

设备检查通过公开 ACLNN 入口，覆盖 1/2/4 行冷启动、热续用、请求变更、正常与缺槽请求混合、差 1/7 个槽的回收，以及拒绝不完整的有效请求组。三种工况都检查动态 UB 扩展起点、524288／524289 的查找切换边界和 1048576 上界。
