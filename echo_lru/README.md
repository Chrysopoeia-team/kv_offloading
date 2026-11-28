# echo_lru

面向昇腾解码的 KV 缓存规划算子。输入是 indexer 选出的 TopK token 索引；算子在每个请求的驻留缓存池（LRU）里查出各 token 的槽位，输出去重后的待搬入清单，并原地更新池状态。框架拿到输出后即可调度 KV 搬运。

## 为什么需要这个算子

稀疏注意力每步只使用 TopK 个 KV。KV 驻留在设备外时，每一步、每一层都要先把选中的 KV 搬上设备，于是需要一个确定的答案：哪些 token 要搬、放到哪个槽、哪些旧槽让位。这个答案由每个请求的 LRU 池决定，必须与 KV 的真实驻留位置逐槽一致。

规划本身处在 decode 的关键路径上，所以有两个要求：延迟低；能和上游 indexer 在同一条流上零拷贝衔接。

设计上沿用 [Echo（OSDI'26）](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) 的思路：把 KV offload 的管理面做成独立的阶段，为之后用其他计算掩藏它留出空间。本目录据此给出两条路径，外加一个对输入没有排序要求的回退版本。

## 三种变体

| 目录 | 定位 | TopK 输入 | 每次调用的 kernel | 容量（槽 / 请求） | 位置数 P |
|---|---|---|---|---|---|
| [`plan/`](plan/INTEGRATION.md) | 当前主要交付版本 | 每行有效前缀升序 | k1、k2，依次在调用流上 | 1–16384 | 1、2、4、8 |
| [`plan_deferred/`](plan_deferred/INTEGRATION.md) | 储备版本，为后续掩藏准备 | 每行有效前缀升序 | k1、k2′ 在调用流上；k3 默认在侧流 | 1–8192 | 1、2、4 |
| [`plan_unordered/`](plan_unordered/INTEGRATION.md) | 回退保证 | 任意顺序 | k1、k2，依次在调用流上 | 1–8192 | 实测 1–4 |

P = MTP + 1，是一个请求在一步内预测的位置数；同一请求的所有位置共用一个池。实测均取 topk = 2048、容量 = P × 2048。

三个变体注册的是同一个 torch 算子名，一个进程只能加载其中一个。

## 工作原理

以 `plan` 为例，一次调用是同一条流上的两个 kernel：

1. **k1，查找与分类**（`echo_lru_kernel.h`）。每个 block 负责一段连续的请求。对每个请求，先把 P 行升序 TopK 合成一份升序的去重 token 表（P = 1 直接建表，P > 1 用硬件归并排序后去重）；再沿驻留槽的 LRU 顺序扫描，把每个槽的 token 在表里二分，得到本步命中的槽位；最后写出每个查询的槽位、升序的 miss 清单，以及 victim / survivor / hit 掩码。
2. **k2，合并、准入与簿记**（`echo_v2_kernel2.cpp`）。按 LRU 顺序淘汰最旧的未命中槽，miss 按 token 升序从空闲栈取槽准入，更新 htd / dth，重建 LRU 顺序，并补齐 miss 清单的输出格式。

`plan_deferred` 把第 2 步拆开：k2′ 做完本步的输出（槽位、miss 清单）就返回，只影响下一步的簿记交给 k3。k3 默认发到侧流，可以和后面的计算重叠；`echo_lru_join` 在需要时让调用流等待 k3。

`plan_unordered` 不要求输入升序，miss 按首次出现的顺序准入；代价是不支持 P = 8 和更大的容量。高性能的去重与扫描依赖升序输入，所以它基于较早的实现。

## 上游 QLI

算子的输入是上游 indexer（QLI）给出的 TopK 索引。QLI 的源码**不在本仓库里**，需要单独获取：

| 用途 | 版本 | 源码 |
|---|---|---|
| `plan`、`plan_deferred` 的输入（升序 TopK） | 保序 QLI V1，算子 `quant_lightning_indexer_ordered` | [`sparse/ops-transformer`，分支 `wzj_qli_order_fused`](https://gitcode.com/sparse/ops-transformer/tree/wzj_qli_order_fused/attention/quant_lightning_indexer_ordered) |
| `plan_unordered` 的输入；测试里的原版基线 | 原版 QLI V1，`npu_quant_lightning_indexer` | torch_npu 自带，不需要单独构建 |

保序 QLI 的构建和安装方法见 [tests/README.md](tests/README.md)。

## 构建

已验证环境：Linux aarch64、Ascend950DT、CANN 9.2.1、PyTorch 2.10.0、torch_npu 2.10.0.post7。先初始化 CANN 开发环境（提供 bisheng），并在已安装 torch / torch_npu 的 Python 下执行：

```bash
bash plan/build.sh             # -> plan/build/libecho_lru_plan.so
bash plan_deferred/build.sh    # -> plan_deferred/build/libecho_lru_plan_deferred.so
bash plan_unordered/build.sh   # -> plan_unordered/build/libecho_lru_plan_unordered.so
```

每个脚本用 bisheng 编译两个设备编译单元，再用系统 C++ 编译器把宿主侧的 torch 算子和它们链接成一个库。编译选项与测量结果时使用的完全相同，写在脚本里。

```python
import torch
import torch_npu

torch.ops.load_library("plan/build/libecho_lru_plan.so")
aiv = torch.ops.sparse_kv_plan_op.echo_lru_core_count()   # 当前设备的向量核数
```

接口见各变体的 `INTEGRATION.md`。

## 测试

`tests/` 是一套联合测试脚本：对每个 (batch, 序列长度, MTP, 缓存状态) 先用上游 QLI 生成 TopK，再逐项核对算子输出和全部状态张量与 CPU 参考是否一致，通过后才采集 kernel 耗时。运行方法见 [tests/README.md](tests/README.md)。

## 结果

[RESULTS.md](RESULTS.md) 给出本版本三个变体的设备实测数据：测量环境、口径、逐形状的 plan 设备耗时，以及正确性覆盖范围。每个版本的改动见 [CHANGELOG.md](CHANGELOG.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `plan/src/` | 主交付版本的源码 |
| `plan_deferred/src/` | 储备版本的源码（多一个 k3） |
| `plan_unordered/src/` | 无序输入回退版本的源码 |
| `plan*/src/echo_v2_sort_tu_launch.h` | 早期方案遗留的声明头，被 `echo_v2_kernel2_launch.h` 引用。对应的 kernel 已不再发射，它的源文件不在仓库里 |
| `*/build.sh`、`*/INTEGRATION.md` | 构建脚本和接口说明 |
| `tests/` | 联合测试脚本和 CPU 参考 |
| `RESULTS.md`、`CHANGELOG.md` | 实测结果、版本说明 |
