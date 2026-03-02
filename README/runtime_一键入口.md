# Runtime 一键入口（方案B默认链路，2026-02-28）

## 0. 默认原则
- 默认入口统一使用 `scripts/memory_indexer/runtime_v2/`，不再要求手改 legacy 脚本参数。
- 默认评测口径是 `S-only`；`mix(auto)` 仅作为可选附加项。
- 默认 `tau` 为常数 `0.1`（`--no-bipartite-learnable-tau`），并写入权重 meta 与训练日志。

## 1. 先做数据链路体检（必跑）
```bat
cmd /c "python -m scripts.memory_indexer.runtime_v2.doctor_data_chain --config configs/default_v2_bipartite.yaml"
```

输出位置：
- `runs/rt_v2/doctor/<timestamp>/doctor.report.json`

## 2. 训练（listwise + bipartite，默认30轮，全量query）
```bat
cmd /c "python -m scripts.memory_indexer.runtime_v2.train_listwise_bipartite_full --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users"
```

默认行为：
- 命中 `eval_cache_users.jsonl` + `memory_cache_users.jsonl` 时，跳过 HF/E5 初始化。
- 输出并覆盖：`data/ModelWeights/listwise_bipartite_reranker.pt`
- run 目录：`runs/rt_v2/listwise_bipartite_train/<timestamp>/train/`

## 3. 快速评测（默认 S-only）
```bat
cmd /c "python -m scripts.memory_indexer.runtime_v2.eval_followup_plus_chat_bipartite_quick --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users"
```

可选：增加 `mix(auto)` 对照
```bat
cmd /c "python -m scripts.memory_indexer.runtime_v2.eval_followup_plus_chat_bipartite_quick --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users --include-mix"
```

## 4. tiny vs bipartite 同口径对照
```bat
cmd /c "python -m scripts.memory_indexer.runtime_v2.compare_tiny_vs_bipartite --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users --max-eval-queries 1000 --eval-sample-mode random --eval-sample-seed 11"
```

## 5. 旧脚本说明（legacy）
- `scripts/memory_indexer/runtime/` 下脚本保留用于回归/应急，不再作为默认入口。
- 新增能力与默认值（model_family/tau/S-only/doctor）以 `runtime_v2` 为准。
