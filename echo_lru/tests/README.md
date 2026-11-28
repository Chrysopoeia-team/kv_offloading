# 联合测试

这套脚本对 `plan`、`plan_deferred`、`plan_unordered` 做两件事：

1. **正确性。** 对每个 (batch, 序列长度, MTP, 缓存状态)，先让上游 QLI 生成 TopK，再调用算子，把 6 个状态张量和 4 个输出与 CPU 参考逐项比对。任何一个用例不一致都会记为失败。
2. **性能。** 校验通过后，用 torch_npu profiler 采集链路里各 kernel 的耗时，输出 plan 设备耗时 P50、完整链设备耗时，以及相对原版 QLI 的增量。

## 前置条件

- 已按 [README](../README.md) 构建好要测的库。
- 已初始化 CANN 运行环境，`npu-smi info` 能看到空闲卡。
- `plan` 和 `plan_deferred` 消费升序的 TopK，需要保序 QLI V1 算子包（[`sparse/ops-transformer`，分支 `wzj_qli_order_fused`](https://gitcode.com/sparse/ops-transformer/tree/wzj_qli_order_fused/attention/quant_lightning_indexer_ordered)，第三方，不在本仓库里）。在它的检出里构建并安装：

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=quant_lightning_indexer_ordered --vendor_name=qliorder -j64
  ./build_out/cann-ops-transformer-qliorder_linux-aarch64.run --install-path=<INSTALL_DIR>
  bash build.sh --torch_extension --soc=ascend950 --ops=quant_lightning_indexer_ordered --vendor_name=qliorder
  pip install --force-reinstall --no-deps build_out/cann_ops_transformer_qliorder-*.whl
  export KV_ORDER_ENV=<INSTALL_DIR>/vendors/qliorder_transformer/bin/set_env.bash
  ```

  `plan_unordered` 用的是 torch_npu 自带的原版 QLI，不需要这个包；但标准矩阵会同时测原版和保序 QLI 两个基线，所以 orchestrator 仍然要求它存在。

## 运行

```bash
cd tests
bash run_standard.sh                 # 2 号卡，标准矩阵，三个变体
bash run_standard.sh 3               # 3 号卡
bash run_standard.sh 3 --methods plan          # 只测主交付版本
bash run_standard.sh 3 --samples 5 --profile-steps 1   # 快速检查
```

标准矩阵：GLM-5.2 和构造两种数据 × batch 8、16 × 序列长度 102,400、358,400 × MTP 0、1、3 × 冷重置和热满池约 90% 命中。完整跑一遍约 35–40 分钟，大部分时间花在解析 profiler 的 CSV 上。

结果写到 `tests/results/`：`unified_glm.{json,md}` 和 `unified_constructed.{json,md}`。先看每张表前的 `golden` 是否全部通过，再看耗时。

## 单独跑一个用例

```bash
source $KV_ORDER_ENV
python3 bench_unified.py --worker --kind plan --method plan \
  --data constructed --device 3 --batches 8 --seqs 102400 --mtps 0 \
  --scenarios cold_reset --samples 2 --profile-steps 1 --part-out /tmp/one.json
```

`/tmp/one.json` 里的 `cases[].golden.ok` 为 true 才说明这个用例通过。

## 说明

- 三个库注册的是同一个算子名，所以 orchestrator 为每个候选单独起一个 worker 进程。
- `plan_deferred` 的 k3 默认走侧流。测试脚本把 `ECHO_PATH2_ASYNC` 设为 0，让 k3 回到调用流上，这样三个变体的耗时是同一口径。
- CPU 参考在 `ref/` 下，是只读文件：`plan_reference.py` 按 token 升序准入，对应 `plan` 和 `plan_deferred`；`plan_unordered_reference.py` 按首次出现的顺序准入。
- 输入模板缓存在 `/dev/shm/union_kv_fixtures/`，profiler 的临时文件在 `/dev/shm/unionprof.*`，可以用 `UNION_PROFILE_ROOT` 改位置。
- 耗时是共享机器上的读数，重复运行会有 ±2–5 µs 的漂移。
