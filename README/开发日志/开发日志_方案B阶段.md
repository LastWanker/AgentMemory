【维护规则】本日志仅允许“追加补充”，禁止删除或改写既有内容。

# 开发日志（方案B阶段）
更新时间：2026-02-27

## 2026-03-01

- 为 `_V3` 新增 `readme/阶段性计划书_主链补全阶段.md`，把当前阶段从“demo 骨架”明确切到“主链补全、可训练、可评测”。
- 完成 `_V3` 主链中段实现：新增训练样本生成、torch listwise reranker、离线评测模块与 full runtime 入口。
- 新增脚本：
  - `_V3/scripts/build_training_samples.py`
  - `_V3/scripts/train_reranker.py`
  - `_V3/scripts/eval_offline.py`
  - `_V3/scripts/doctor_data_chain.py`
  - `_V3/scripts/runtime_full_pipeline.py`
- 新增源码：
  - `_V3/src/agentmemory_v3/retrieval/feature_builder.py`
  - `_V3/src/agentmemory_v3/training/common.py`
  - `_V3/src/agentmemory_v3/training/trainer.py`
  - `_V3/src/agentmemory_v3/models/reranker.py`
  - `_V3/src/agentmemory_v3/evaluation/offline_eval.py`
- 修改 `_V3` 检索器，使其在存在 `data/V3/models/reranker.pt` 时自动加载训练后的 reranker；无权重时则回退到启发式精排。
- 修改 `_V3/scripts/build_slots.py`，支持断点续跑和分批 flush，避免全量槽抽取中途失败后必须重跑。
- 当前已完成 smoke 验证：
  - `cmd /c "python _V3\scripts\build_training_samples.py --max-queries 200"`
  - `cmd /c "python _V3\scripts\train_reranker.py --epochs 2 --batch-size 16"`
  - `cmd /c "python _V3\scripts\eval_offline.py --split test --max-queries 200"`
  - 结果示例：`RetrievalRecall@N=0.7551`，`MRR=0.8581`，`Top1=0.8300`
- 继续推进主链对齐：
  - 新增 `_V3/scripts/build_query_slots.py`
  - 新增 `slot_bipartite_8x8` 大型神经精排器
  - query slot 已接入训练与离线评测
  - 当前检索侧采用 “bipartite + 特征兜底” 过渡推理
- 当前断点续跑进度：
  - `data/V3/processed/memory_slots.jsonl` 已推进到 500+ 条
  - `data/V3/processed/query_slots.jsonl` 已推进到 600 条
- 当前阶段性验证：
  - `cmd /c "python _V3\scripts\train_reranker.py --epochs 4 --batch-size 8 --device cpu"`
  - `cmd /c "python _V3\scripts\eval_offline.py --split test --max-queries 200"`
  - 在部分正式 slot 条件下，`runtime_retrieve_demo` 已重新命中 `n001`
- 继续推进到全量 slot：
  - `query_slots.jsonl` 已全量跑满：4806
  - `memory_slots.jsonl` 已全量跑满：3728
  - `query slot` 全量为 `deepseek`
  - `memory slot` 统计为 `deepseek=3722 / fallback=26`
- 新增维护脚本：
  - `_V3/scripts/dedupe_jsonl_by_key.py`
- 正式训练已提升到 15 轮：
  - `cmd /c "python _V3\scripts\train_reranker.py --epochs 15 --batch-size 8 --device cpu"`
  - 当前 full test 结果：`RetrievalRecall@N=0.8154`，`MRR=0.5475`，`Top1=0.4908`
- 训练入口已改为自动选设备：若当前 Python 环境支持 CUDA，则优先走 GPU，并使用更大的默认 batch。
- 环境检查结论：
  - `Anaconda python` 中的 `torch` 是 `2.3.0+cpu`
  - 仓库 `.venv` 中的 `torch` 是 `2.5.1+cu121`
  - 因此 V3 训练/评测今后应优先使用 `.venv\Scripts\python.exe`
- 已在 `.venv + cuda` 下重跑 full train/eval：
  - `cmd /c ".venv\Scripts\python.exe _V3\scripts\train_reranker.py --epochs 15 --device auto"`
  - `cmd /c ".venv\Scripts\python.exe _V3\scripts\eval_offline.py --split test"`
  - 当前结果：`RetrievalRecall@N=0.8083`，`MRR=0.5509`，`Top1=0.4949`

## 1. 阶段目标
- 目标：把 reranker 默认链路切到 bipartite-align-transformer（listwise 主链路），并控制改造风险。
- 范围：先解决数据链路与默认入口问题，再推进模型升级；当前阶段不做重型性能优化。

## 2. 执行原则（沿用 tinyreranker 阶段教训）
- 先保证数据与缓存正确，再看模型指标。
- 正式缓存与烟测缓存硬隔离（`users` / `users_smoke`）。
- 默认参数必须可解释，避免隐式截断与隐式重建。
- 评测必须同时输出 `S-only` 与 `mix(auto)`，分离模型贡献和融合影响。

## 2026-02-27（方案B计划立项与目录切换）

### 本次变更摘要
- 新增计划书：`README/问题报告＆计划书/计划书_2026-02-27_B方案默认链路切换与防踩坑改造.md`
- 计划明确四类重点：
  1) 数据链路体检与配置冻结（P0）
  2) 新 scorer 接入（P1）
  3) 训练协议统一到 listwise 主链路（P2）
  4) runtime_v2 一键入口与默认切换（P3/P4）

### 关键约束
- 新链路必须成为默认入口；legacy 仅作为回滚通道。
- 当前阶段明确不做重型性能投入，避免在“优化时间”上继续埋坑。
- 所有覆盖动作（权重/缓存）必须备份并可追溯来源 run。

### 下一步执行顺序
1. 先实现 `doctor_data_chain.py` 与 `default_v2_bipartite.yaml`。  
2. 再接入 `learned_scorer_bipartite.py` 与 scorer factory。  
3. 然后升级训练脚本默认模型族并新建 runtime_v2 一键脚本。 
4. 最后做固定口径对照评测并回写问题报告与本日志。 

## 2026-02-28（方案B按计划书完整执行）

### 执行批次
- 体检：`python -m scripts.memory_indexer.runtime_v2.doctor_data_chain --config configs/default_v2_bipartite.yaml`
- 训练：`python -m scripts.memory_indexer.runtime_v2.train_listwise_bipartite_full --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users`
- 评测：`python -m scripts.memory_indexer.runtime_v2.eval_followup_plus_chat_bipartite_quick --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users`
- 对照：`python -m scripts.memory_indexer.runtime_v2.compare_tiny_vs_bipartite --config configs/default_v2_bipartite.yaml --dataset followup_plus_chat --cache-alias users --max-eval-queries 1000 --eval-sample-mode random --eval-sample-seed 11`

### 本轮确认
- doctor = PASS；正式缓存 `users` 完整可用，`users_smoke` 未污染正式链路。
- 训练走缓存命中路径，日志显示已跳过 HF/E5 初始化。
- 默认权重已覆盖为 `data/ModelWeights/listwise_bipartite_reranker.pt`。
- `tau` 口径固定为常数 `0.1`，并写入训练日志与权重 meta（`bipartite_learnable_tau=false`）。
- 默认评测组为 `S-only`，`mix(auto)` 仅可选。

### 指标记录（S-only, half_hard, 1000随机样本）
- bipartite：`Recall@k=0.409`，`Recall@P=0.386`，`CoarseRankRecall@k=0.402`，`CoarseRankRecall@P=0.387`，`Gain@k=0.007`，`Gain@P=-0.001`，`MRR=0.670`，`Top1=0.616`。
- tiny 对照：`Recall@k=0.394`，`Recall@P=0.370`，`Gain@k=-0.008`，`Gain@P=-0.016`。

### 运行产物
- doctor：`runs/rt_v2/doctor/20260228_015910/doctor.report.json`
- 训练：`runs/rt_v2/listwise_bipartite_train/20260228_023027/train/`
- quick eval：`runs/rt_v2/followup_plus_chat_eval/20260228_030426/bipartite/eval/`
- 对照：`runs/rt_v2/compare_tiny_vs_bipartite/20260228_030655/`

### 后续观察点
- bipartite 评分耗时明显高于 tiny（本次约 85ms vs 42ms / query），后续需要独立做吞吐优化。
- `Gain@P` 接近0但未稳定转正，优先继续做训练样本与loss权重校准。

## 2026-02-28（_V3 隔离环境立项与讨论稿落地）

### 本次变更摘要
- 根目录新增 `_V3/`，作为与母项目隔离的 V3 工作区骨架。
- 已创建空目录：`_V3/src/`、`_V3/scripts/`、`_V3/configs/`、`_V3/runs/`、`_V3/readme/`。
- 新增文档：
  - `_V3/README.md`
  - `_V3/readme/问题分析.md`
  - `_V3/readme/计划书.md`

### 当前约束
- 本轮只做讨论、结构预留和计划冻结，不写 V3 实现代码。
- V3 与母项目代码、脚本、配置、run 目录完全隔离；仅 `data/` 允许共享物理存储。
- V3 主线限定为 `hybrid coarse recall -> cluster expansion -> reranker`，不再保留旧的多通道过渡形态。

### 下一步建议
1. 先确认 `_V3/readme/问题分析.md` 末尾的待拍板项。  
2. 再按 `_V3/readme/计划书.md` 的 P1 开始落数据契约和目录内首批脚本。  
3. 代码实现阶段优先做数据适配与 hybrid 粗召回，不先碰 Agent/UI。  

## 2026-02-28（V3 批注吸收与 `_chat` 架构落地）

### 本次变更摘要
- 按 `_V3/readme/问题分析.md` 中 `【】` 批注修订了 `_V3/readme/计划书.md`。
- 新增 `_V3/readme/实现架构.md`，明确 V3 与 `_chat` 的双工程结构。
- 新增 `_V3/configs/` 下的默认配置模板。
- 新增 `_V3/scripts/` 9 个脚本占位入口，用于冻结实现边界和后续命令名。
- 新增 `_V3/src/agentmemory_v3/` 骨架与基础数据契约文件。

### 批注吸收结果
- cluster 扩召上限从 `200` 收敛到 `150`。
- reranker 保留 V2 的 `8x8` 精排思路。
- 8 个输入向量默认改为 8 槽位 embedding 主线路。
- token 备选方案保留为 `top-32 -> MMR 选 8`。
- query augmentation 第一版保持现有方案，不新增 perspective 分支。
- 本地聊天器改为 Python 先行，位置固定为根目录 `_chat/`。

## 2026-02-28（V3 检索链路与 `_chat` 本地聊天器最小可运行版）

### 本次落地
- `_V3/scripts/build_dataset.py`
  - 已可将 `data/Processed/memory_followup_plus_chat.jsonl` 与 `eval_followup_plus_chat.jsonl` 归一化为 `data/V3/processed/{memory,query,cluster}.jsonl`
- `_V3/scripts/build_slots.py`
  - 已支持读取 `data/_secrets/deepseek.env`
  - 已可通过 DeepSeek API 执行 8 槽 IE 抽取
  - 保留 fallback，仅在 API 失败时兜底
- `_V3/scripts/build_hybrid_index.py`
  - 已可离线构建 dense + BM25 索引
  - 已把槽位文本向量单独落到 `slot_vectors.npz`
- `_V3/scripts/export_chat_bundle.py`
  - 已可导出 `_chat` 直接使用的记忆 bundle
- `_V3/scripts/serve_retriever.py`
  - 已可启动 FastAPI retriever service
- `_chat/`
  - 已可读取 `data/_secrets/deepseek.env`
  - 已能会话落盘、展示参考记忆，并调用 DeepSeek API 生成回答

### 速度与链路约束
- V3 检索采用离线建索引，在线只做查询与重排，不在请求时重新编码整库。
- dense 侧是离线 `TF-IDF + SVD` 矩阵化索引。
- keyword 侧是离线 BM25。
- chat 请求链路默认优先走 HTTP retriever；服务不可用时回退到本地 `_V3` retriever；最后才退到 bundle overlap。

### Smoke 结果
- `build_dataset.py` 已实际跑通：
  - memory=`3728`
  - query=`4806`
  - cluster=`1232`
- DeepSeek 单条槽位抽取已跑通，返回 `slot_source=deepseek`。
- retriever HTTP smoke 已命中 `n001`（黄山/无人机/消防员事件）。
- `_chat` 真实对话 smoke 已能回答该记忆，并正确展示参考记忆 top1=`n001`。

## 2026-02-28（可直接运行入口补齐）

### 本次补齐
- `_chat/app.py`
  - 新增 `main()`，现在可直接：
    - `python _chat\app.py`
- 新增：
  - `_V3/scripts/runtime_prepare_demo.py`
  - `_V3/scripts/runtime_retrieve_demo.py`
- 更新：
  - `_V3/README.md`
  - `_chat/README.md`

### 当前最短使用命令
1. 准备 demo 数据：`cmd /c "python _V3\scripts\runtime_prepare_demo.py --profile smoke"`  
2. 看单条检索：`cmd /c "python _V3\scripts\runtime_retrieve_demo.py --query 黄山 无人机 消防员 罚款 --top-k 3"`  
3. 启动聊天器：`cmd /c "python _chat\app.py"`  

### 启动验证
- 已验证 `python _chat\app.py` 会拉起 Uvicorn，并监听 `http://127.0.0.1:7860`。
