# plan_deferred 接入说明

`plan_deferred` 是储备版本，为后续把规划的开销掩藏到其他计算后面做准备。它的张量、输入输出、处理规则和宿主参数与 [`plan`](../plan/INTEGRATION.md) 完全相同，下面只写差异。

输入同样来自保序 QLI V1（[`sparse/ops-transformer`，分支 `wzj_qli_order_fused`](https://gitcode.com/sparse/ops-transformer/tree/wzj_qli_order_fused/attention/quant_lightning_indexer_ordered)）。

## 与 plan 的差异

| | plan | plan_deferred |
|---|---|---|
| 位置数 P | 1、2、4、8 | 1、2、4 |
| 容量 C | 1–16384 | 1–8192 |
| decode 调用发射的 kernel | k1、k2 | k1、k2′、k3 |
| 簿记（重建 `lru`、清理 `dth`、更新 `fifo` / `avail`） | 在 k2 内完成 | k2′ 返回后由 k3 完成 |

工作区按 `plan` 说明里的 `workspace_elements()` 分配即可：`plan` 的需求比本版本多每请求一个元素，是安全上界。

## 两个阶段

k1 和 k2′ 在调用流上依次执行。k2′ 完成时，本步 attention 需要的东西都已经就绪：`current_slots`、`miss_tokens`、`miss_slots`、`miss_count`，以及准入 token 的 `htd` / `dth`。

k3 处理只有**下一步** decode 才会读的状态：重建 `lru`，把淘汰后没有被复用的槽的 `dth` 置为 INT32_MAX，更新 `fifo` 和 `avail`。它默认发到一条侧流上，因此可以和调用流上后面的计算重叠。

k3 必须先于下一步的 k1 完成。算子在宿主侧保证这一点：每次 `echo_lru` 调用开始时，调用流先等待上一次的 k3；`echo_lru_init` 同样如此。

## 开关

两个环境变量在库加载后首次调用时读取一次。

| 变量 | 默认 | 含义 |
|---|---|---|
| `ECHO_PATH2_ASYNC` | 开 | `0` 时 k3 改发到调用流，三个 kernel 依次执行 |
| `ECHO_PATH2_CAPTURE` | 关 | 捕获 NPU 图时默认自动改走调用流；设为 `1` 则在图内保持侧流（原型行为） |

侧流模式不能被图捕获。需要捕获图时，对每个形状先在图外预热一次，再捕获；捕获过程中 k3 走调用流。

## 收尾

```python
torch.ops.sparse_kv_plan_op.echo_lru_join()
```

让调用流等待最近一次发射的 k3。如果下一段工作在另一条流上、并且会读这些状态张量，或者准备释放它们，先调用它。

## 性能口径

侧流模式下，调用流上的规划开销是 k1 + k2′；k3 的耗时被掩藏。`ECHO_PATH2_ASYNC=0` 时三个 kernel 都在调用流上，总耗时是 k1 + k2′ + k3。两种口径的数字见 [RESULTS.md](../RESULTS.md)，不要把边界口径和 `plan` 的全含口径直接比较。
