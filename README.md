# kv_offloading

## 算子

| 目录 | 说明 |
|---|---|
| [`echo_lru/`](echo_lru/README.md) | KV 缓存规划算子：在每请求的 LRU 驻留池里查找 TopK token 的槽位，输出去重后的待搬入清单并更新池状态 |
