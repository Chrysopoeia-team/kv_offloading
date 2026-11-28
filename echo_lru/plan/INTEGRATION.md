# plan 接入说明

`plan` 是当前的主要交付版本。输入每个位置的升序 TopK，输出各位置的槽位和去重后的待搬入清单，并原地维护每个请求的 LRU 池。

## 输入和编号

上游是保序 QLI V1（算子 `quant_lightning_indexer_ordered`，[`sparse/ops-transformer`，分支 `wzj_qli_order_fused`](https://gitcode.com/sparse/ops-transformer/tree/wzj_qli_order_fused/attention/quant_lightning_indexer_ordered)，源码不在本仓库里）。它输出每个位置的 TopK 索引，行内升序、尾部用 −1 填充。

记请求数为 `B`，位置数 `P = MTP + 1`，每位置 TopK 数为 `K`，`htd` 的行宽为 `L`（池内可容纳的逻辑序列长度），每请求每层的槽位容量为 `C`。实测取 `K = 2048`、`C = P × 2048`。

- `pos` 的形状是 `[B × P, K]`，第 `r × P + p` 行是请求 `r` 在位置 `p` 的 TopK。**每行的有效前缀必须按 token 编号升序且互不相同。**
- token 编号从 1 开始：`pos = 上游索引 + 1`。编号 0 是保留值，`pos` 为 0 的项按 padding 处理，输出槽位 0。上游用 −1 填充尾部时，加一后自然变成 0。
- 同一请求的不同位置允许出现相同的 token，它们共享同一个槽位。
- 支持范围：`P ∈ {1, 2, 4, 8}`，`K ≤ 65535`，`1 ≤ C ≤ 16384`；`P = 8` 时要求 `C ≥ P × K`，`C > 8192` 时要求每个 block 只处理一个请求（见下文）。

## 每层分配的七个状态张量

每层独立分配，常驻 GM、原地更新。一个请求在各轮调用中对应同一行。`C` 固定时，改变 `P` 可以复用全部七张表。

| 形参 | dtype | 形状 | 含义 |
|---|---|---|---|
| `htd` | int32 | [B, L] | token 到槽位。`htd[:, 0] = 0`，空的 token 为 INT32_MAX |
| `dth` | int32 | [B, C + 1] | 槽位到 token。槽位从 1 开始，空槽为 INT32_MAX |
| `lru` | int32 | [B, C + 8] | 占用槽从最旧到最新排列，有效长度为 `C − avail`，其余为 0。行尾 8 个元素是对齐用的 |
| `free_slots` | int32 | [B, C + min(K, C)] | 空闲槽位栈，栈长为 `avail`；本步淘汰的槽先压入这里 |
| `avail` | int32 | [B] | 各请求的空闲槽数 |
| `fifo` | int32 | [B] | 每次调用加 2 的计数，重置当轮结束时为 3 |
| `fresh` | uint8 | [B] | 行处于 init 态的标志。由 `echo_lru_init` 置位，decode 首次装载后清零 |

## 输入、输出及工作区

所有张量位于同一 NPU 且连续。接口原地更新七张状态表和三个输出，Python 返回值为 `None`。

| 形参 | dtype | 形状 | 含义 |
|---|---|---|---|
| `pos` | int32 | [B×P, K] | 上游 TopK 索引加一 |
| `spl` | int32 | [B] | 投机前缀长度。`spec_enabled = True` 时，调用开始前把池内编号不小于 `spl` 的 token 移出池 |
| `reset_mask` | uint8 | [B] | 1 表示本次调用开始时把该行清为空池，0 表示沿用上一轮状态 |
| `current_slots` | int32 | [B×P, K] | 与 `pos` 逐项对应的槽位。`pos` 为 0 时为 0；池放不下而未获槽位的有效 token 为 INT32_MAX |
| `miss_count` | int32 | [B] | 本轮实际准入的不同 token 数 |
| `miss_tokens` | int32 | [B, P×K] | 前 `miss_count` 项为升序、去重的准入 token，其余为 −1 |
| `miss_slots` | int32 | [B, P×K] | 与 `miss_tokens` 一一对应的槽位，其余为 −1 |
| `workspace` | int32 | 一维，不少于 `workspace_elements()` 的返回值 | 两个 kernel 交接用，内容只在本次调用内有效 |

`workspace_elements()` 的算法在 `src/echo_lru_launch.h`（`EchoWorkspaceElements`），下文示例里有同等的 Python 写法。

## 每个请求的处理规则

1. `reset_mask = 1` 时先清池：`htd` 全为 INT32_MAX（`htd[:, 0]` 为 0），`dth` 全为 INT32_MAX，`lru` 为 0，`free_slots = [C, C − 1, …, 1, 0, …]`，`avail = C`，`fifo` 置 1。
2. 判定：本请求的全部 `P × K` 个查询读 `htd`，读到槽位即命中，读到 INT32_MAX 即缺失。
3. 淘汰：需要腾出的槽位数是 `max(原始缺失数 − avail, 0)`，原始缺失数跨位置不去重。从本步未命中的占用槽里按 LRU 顺序取最旧的一些，最多取完。
4. 准入：不同的缺失 token **按 token 编号升序**从空闲栈顶取槽。放不下的不分配：`current_slots` 保持 INT32_MAX，也不进入 miss 清单。
5. 同一 token 出现在多个位置时共享同一个槽位，只准入一次。
6. 重建 LRU：幸存者（旧序）在前，其次是本步命中的槽（旧序），最后是新准入的槽（准入序）；其余补 0。
7. `miss_count` 为准入数，`fifo` 加 2，`avail` 为准入后的空闲数。

淘汰数按未去重的缺失数计算，是一种保守的策略：缺失较多时会多腾出一些槽，它们留在空闲栈上供后续步骤使用。

## 宿主侧 launch

```python
plan = torch.ops.sparse_kv_plan_op.echo_lru
plan(pos, spl, reset_mask, htd, dth, lru, free_slots, avail, fifo, fresh,
     current_slots, miss_count, miss_tokens, miss_slots, workspace, blocks, spec_enabled)
```

框架只需调用一次。`blocks` 是 block 数：每个 block 处理连续的 `ceil(B / blocks)` 个请求，要求 `1 ≤ blocks ≤ B` 且不超过设备的向量核数。`B ≤ 32` 时取 `blocks = min(B, echo_lru_core_count())`，每个请求一个 block；`C > 8192` 时必须 `blocks = B`。

launch 参数由张量形状推导。它的 device 拷贝按（设备，内容）缓存，形状不变时不发生 H2D，也不查询设备属性。两个 kernel 依次发到当前 PyTorch NPU 流上，中间没有宿主同步。

## 首次调用

新请求可以直接用 `reset_mask = 1` 开始：清池在 decode 调用内完成，`htd` 整行都会被重写，这部分耗时随 `L` 增长。如果希望把这笔开销移出 decode，可以改用 `echo_lru_init`：

```python
torch.ops.sparse_kv_plan_op.echo_lru_init(init_mask, htd, dth, lru, free_slots,
                                          avail, fifo, fresh, blocks)
```

它在 decode 之外把 `init_mask` 选中的行初始化为空池并置 `fresh = 1`，这些行的第一次 decode 调用不再清 `htd`。

```python
B, P, K, L, C = 8, 4, 2048, 262_144, 8192
n = B * P
dev = "npu:0"
i32, u8 = torch.int32, torch.uint8

def alloc(shape, dtype=i32):
    return torch.empty(shape, dtype=dtype, device=dev)

def workspace_elements(requests, tokens, topk, capacity, blocks):
    o = requests * (capacity + 1) + 5 * requests
    o += 3 * tokens * topk + requests
    o = (o + 7) // 8 * 8
    o += requests * 3 * ((capacity + 31) // 32) + requests * 8 + blocks * 2
    return (o + 1) // 2 * 2

pos = ordered_topk_index.view(n, K) + 1          # 上游提供，行内升序
spl = torch.full((B,), L, dtype=i32, device=dev)  # 不使用投机后缀作废
reset_mask = torch.ones(B, dtype=u8, device=dev)
htd, dth, lru = alloc((B, L)), alloc((B, C + 1)), alloc((B, C + 8))
free_slots = alloc((B, C + K))
avail, fifo, fresh = alloc((B,)), alloc((B,)), alloc((B,), u8)
current_slots, miss_count = alloc((n, K)), alloc((B,))
miss_tokens, miss_slots = alloc((B, P * K)), alloc((B, P * K))
workspace = alloc((workspace_elements(B, n, K, C, B),))
blocks = min(B, torch.ops.sparse_kv_plan_op.echo_lru_core_count())

torch.ops.sparse_kv_plan_op.echo_lru(
    pos, spl, reset_mask, htd, dth, lru, free_slots, avail, fifo, fresh,
    current_slots, miss_count, miss_tokens, miss_slots, workspace, blocks, False)
reset_mask.zero_()   # 同一条流，后续调用沿用各请求行的状态
```

首次调用也会更新状态。并发调用同一层的不同请求时，应使用独立的工作区和输出。KV 搬运与后续 attention 通过流顺序或事件保证依赖。
