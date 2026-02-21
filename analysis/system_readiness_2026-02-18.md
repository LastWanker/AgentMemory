# 现在到底具不具备从系统优化泥潭里爬出来的时机

## 结论
具备“阶段性脱泥潭”的条件，可以把主精力转向数据建设（尤其是多正例数据），但仍需保留每周一次的轻量系统巡检。

## 依据（这次已落地）
- `run-dir` 产物路径统一：训练/评测/消融都能稳定写入 `runs/...`，并落 `config.snapshot.json`。
- 实验索引落盘：`runs/registry.csv` 已建立，支持按 `run_dir` 幂等更新，避免“跑过但找不到”。
- 最优模型选择工具：新增 `scripts/list_runs.py` 与 `scripts/select_best_run.py`，可快速定位和固化最佳权重。
- cache 污染风险下降：签名包含 dataset/backend/encoder/strategy + 数据指纹（memory/eval 文件 sha1 短哈希），源文件变更后会自动重建缓存。
- 权重覆盖风险下降：`train_reranker --run-dir` 默认仅写 run-dir，不再默认覆盖 `data/ModelWeights/tiny_reranker.pt`。

## 还没有彻底结束的工程项
- 配置项仍分散在多个脚本参数里（已支持 `--config`，但 schema 仍可再收敛）。
- 历史 run 的字段完整性不完全一致（新 registry 会逐步修复）。
- 部分旧缓存文件可能没有 `_meta.cache_signature`，首次清理时需要手动甄别。

## 接下来建议（重心转向数据）
1. 先固定一套 `followup + hf/e5` 训练/评测命令，连续跑 3~5 天，确认指标方差可控。
2. 把主要时间投入到 `positives` 质量和覆盖度，而不是继续加新开关。
3. 每周一次跑 `list_runs + select_best_run + 复现实验`，确认回归不漂移即可。

## 小白执行版（你只要记住这三条）
1. 看最近结果：`python -m scripts.list_runs --dataset followup --encoder-backend hf`
2. 选最佳模型：`python -m scripts.select_best_run --dataset followup --encoder-backend hf --metric Recall@5`
3. 用最佳模型复现评测：`python -m scripts.eval_router --dataset followup --encoder-backend hf --use-learned-scorer --reranker-path runs/best/followup/hf/tiny_reranker.pt --run-dir runs/<timestamp_eval_best>`
