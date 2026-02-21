# 为什么 `run_ablation_matrix.py` 分数会很低（2026-02-18）

## 结论

是的，核心原因基本就是你说的那条：  
之前 `run_ablation_matrix.py` 默认走的是 `simple` backend，而不是你手动跑出高分的 HF/E5 链路。

---

## 证据链

1. 旧版 `run_ablation_matrix.py` 里训练与评测都写死了：
- `TRAIN_ENCODER_BACKEND = "simple"`
- 调用 `eval_router.py` 时传 `--encoder-backend simple`

2. `simple` backend 本质是哈希向量占位实现，语义能力明显弱于 HF/E5。  
所以 baseline 常见在 `0.0x ~ 0.1x` 是合理现象，不代表“系统坏了”。

3. 你手动跑出 `~0.78` 的路径属于另一套配置（HF/E5 + 对应缓存/参数），不能和 simple 结果直接比较。

---

## 为什么会“看起来像冲突”

不只是 backend 不同，还有缓存混用风险：

- 旧流程可能复用不匹配的 `eval_cache`，进一步放大“能跑但不可信”的感觉。
- 现在已做缓存签名隔离（dataset/backend/encoder/strategy/k），可避免跨配置误复用。

---

## 现在怎么避免这个问题

1. `run_ablation_matrix.py` 已支持 `--encoder-backend` 覆盖，并从 `configs/default.yaml` 读取默认 backend。  
2. 你要复现实验时，确保“同一条链路”三处一致：
- train backend
- eval backend
- cache signature（自动按 backend/encoder 分隔）

---

## 建议的实际用法

- 快速 smoke（simple）：
  - `python scripts/run_ablation_matrix.py --encoder-backend simple --configs baseline --seeds 11 --bootstrap 20`

- 对齐你当前主链路（HF/E5）：
  - `python scripts/run_ablation_matrix.py --encoder-backend hf`

如果 HF 本地模型不可用，再加你现有的 HF 离线/在线策略参数到 train/eval 脚本里统一管理。

