# plan_unordered 接入说明

`plan_unordered` 是回退保证版本，用于上游 indexer **不能保证 TopK 升序**的场景。它与 [`plan`](../plan/INTEGRATION.md) 的接口（算子签名、七个状态张量、输入输出、工作区、`blocks` 与 `echo_lru_init` 的用法）完全相同，下面只写差异。

输入可以来自任何 indexer。本仓库实测时用的是 torch_npu 自带的原版 QLI V1（`npu_quant_lightning_indexer`），不需要额外的包。

## 与 plan 的差异

| | plan | plan_unordered |
|---|---|---|
| `pos` 每行的顺序 | 有效前缀必须升序 | 任意顺序 |
| miss 的准入顺序 | token 编号升序 | 在拼接后的查询流里首次出现的顺序 |
| 位置数 P | 1、2、4、8 | 实测 1、2、3、4 |
| 容量 C | 1–16384 | 1–8192 |
| 基础实现 | 去重与驻留扫描 | 逐查询查表，去重放在 k2 |

高性能的去重和扫描依赖升序输入，在无序输入上不能直接复用，所以这个版本基于较早的实现。它的耗时高于 `plan`（见 [RESULTS.md](../RESULTS.md)），只在无法保证升序时使用。

## 语义上的一个差别

准入顺序依赖输入的排列。给定输入时结果是确定的，但同一组 token 以不同的排列输入时，新准入的 token 可能拿到不同的槽号。按 token 升序准入的 `plan` 没有这个现象。

其余规则相同：同一 token 出现在多个位置只准入一次；按 LRU 顺序淘汰本步未命中的最旧槽；池放不下的缺失不分配，`current_slots` 保持 INT32_MAX，也不进入 miss 清单；`miss_tokens` / `miss_slots` 的尾部填 −1。

## 入图

一次调用只有同一条流上的两次 launch，没有侧流，也没有宿主同步。宿主侧缓存了 launch 参数的 device 拷贝，所以可以被捕获：

- 每个形状先在图外预热一次，再捕获。
- 形状变化（`B`、`P`、`K`、`C`、`L` 或张量地址）后重新捕获。原地更新 `pos`、`reset_mask` 和状态张量时可以继续回放同一张图。

## 最小用法

```python
import torch
import torch_npu

torch.ops.load_library("plan_unordered/build/libecho_lru_plan_unordered.so")

B, P, K, L, C = 8, 3, 2048, 262_144, 6144
n = B * P
dev = "npu:0"
i32, u8 = torch.int32, torch.uint8

def alloc(shape, dtype=i32):
    return torch.empty(shape, dtype=dtype, device=dev)

def workspace_elements(requests, tokens, topk, capacity, blocks):
    o = requests * (capacity + 1) + 4 * requests
    o += 3 * tokens * topk + requests
    o = (o + 7) // 8 * 8
    o += requests * 3 * ((capacity + 31) // 32) + requests * 8 + blocks * 2
    return (o + 1) // 2 * 2

pos = topk_index.view(n, K) + 1                    # 上游提供，顺序不限；-1 填充加一后为 0
spl = torch.full((B,), L, dtype=i32, device=dev)
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
