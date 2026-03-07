# 记忆网络 suppressor 全量提纲（地址优先版）

更新时间：2026-03-06
执行状态：P0/P1/P2 已落地为 `v4` 地址版（`mem_network_single_head_addr_v4`）

## 1. 一句话目标

在不改 `coarse + association` 主召回的前提下，把 suppressor 做成可热关闭的单头记忆网络，训练目标只盯 `feedback pair` 命中，做到“命中即硬删除”。

## 2. 强约束（不讨论）

1. 主链路冻结：`_V5/src/agentmemory_v3/retrieval/*` 与 `association/*` 不改。
2. suppressor 串行后处理：`off / mem_network / legacy` 三态保留。
3. mem_network 模式禁用保险丝：不使用 `keep_top_per_lane`、`max_drop_per_lane`。
4. 推理策略固定：`suppress_score >= threshold` 即删除。

## 3. 问题定义（语义统一）

1. 任务定义：`(query, memory_id) -> suppress_score`。
2. 标签语义：`score 高 = 应压制`，`score 低 = 不压制`。
3. `toforget` 与 `unrelated` 在训练中统一为 suppress 正向锚点，不再分头。
4. 不再用“抚平”目标干扰主目标，不追求 rand vs rand 额外约束。

## 4. 新计划总览

1. P0：数据重构为“地址学习优先”。
2. P1：模型改为“显式 memory_id 键 + 1-hop 读写”。
3. P2：训练目标改为“组内分类 + 组内排序”的单任务。
4. P3：评测只做快速阈值面与突变点细扫。
5. P4：上线保持热开关与主链路零侵入。

## 5. P0 数据重构（先做）

脚本：`_V5/scripts/mem_network_suppressor/build_mem_network_dataset.py`

1. 正样本：每条 feedback 锚点 `(q, m_pos)`，`label=1`。
2. 负样本 A：同 `q` 下其它 `m_neg`，优先难负例（向量邻居），再随机补齐。
3. 负样本 B：同 `m_pos` 下其它 `q_neg`，用于打断“只记 memory 不看 query”。
4. 分组单位固定为 query 组：一个组内包含 `1 个正例 + N 个负例`。
5. 输出中保留组 id 与样本权重，供训练直接消费。

## 6. P1 模型结构（地址优先）

位置：`_V5/src/agentmemory_v3/mem_network_suppressor/model.py`

1. 编码器继续复用 `e5-small`（冻结），只训练 suppressor 小网络。
2. 候选 `memory_id` 经过可学习键映射（address key）。
3. 1-hop 记忆读取保留，但核心打分改成“query 状态 vs memory_id 键”的可寻址匹配。
4. 输出单 logit，再过 `sigmoid` 得分。
5. 参数规模建议：
   - `state_dim=256` 起步；
   - `num_hops=1` 固定；
   - 目标总参约 `1M~2M`，先不扩深层。

## 7. P2 训练目标（只服务 feedback 命中）

脚本：`_V5/scripts/mem_network_suppressor/train_mem_network_suppressor.py`

1. 点式目标：组内二分类（正例拉高、负例压低）。
2. 排序目标：同组 `pos > neg` 的 margin 约束。
3. 权重策略：feedback 正例权重显著高于普通负例。
4. 轮次先固定 `epochs=5`，不扩搜索面。
5. 默认设备 `--device auto`，有 CUDA 则走 GPU。
6. artifact 新增：`suppress_memory_ids.json`、`feedback_memory_id_ids.npy`。

## 8. P3 评测与阈值搜索（快速版）

脚本：`_V5/scripts/mem_network_suppressor/eval_mem_network_suppressor.py`

1. 随机样本固定 `random=300`。
2. 粗扫 5 点阈值。
3. 在突变点附近再做 5 点细扫（类二分）。
4. 仅扫 `threshold` 轴，不扫其它配置轴。
5. 评测指标固定：
   - `feedback_pair_removed_rate`
   - `random_top1_changed_rate`
   - `random_top5_set_changed_rate`

## 9. P4 上线策略（零侵入）

1. 入口仍为 `SuppressorAdapter`，不改两路召回。
2. 后端保持可切换：`off / mem_network / legacy`。
3. mem_network 产物目录继续使用 `data/V5/mem_network_suppressor/current`。
4. 若 artifact 缺失或加载失败，自动回落 `off`（不影响主链路）。

## 10. 通过标准（本轮）

1. 功能性：命中阈值即硬删除，且可热关闭。
2. 可解释性：同 query 组内，正例分数显著高于组内负例。
3. 目标区间：
   - `feedback_pair_removed_rate` 优先拉高；
   - `random_top1/top5` 控制在可接受区间（先看趋势，再定最终线）。
4. 结论口径统一：先看“是否出现稳定可用阈值窗口”，再谈进一步优化。

## 11. 执行命令（当前版）

```bat
.venv\Scripts\python.exe _V5\scripts\mem_network_suppressor\build_mem_network_dataset.py --output-dir data/V5/mem_network_suppressor/dataset_v3_addr
.venv\Scripts\python.exe _V5\scripts\mem_network_suppressor\train_mem_network_suppressor.py --data-dir data/V5/mem_network_suppressor/dataset_v3_addr --artifact-dir data/V5/mem_network_suppressor/current --epochs 5 --num-hops 1 --device auto
.venv\Scripts\python.exe _V5\scripts\mem_network_suppressor\eval_mem_network_suppressor.py --artifact-dir data/V5/mem_network_suppressor/current --random-sample-size 300 --thresholds 0.35,0.45,0.55,0.65,0.75 --output-json data/V5/mem_network_suppressor/eval_two_cohort_fast5.json
```

## 12. 本轮不做的事

1. 不加新门控规则。
2. 不恢复双头。
3. 不改主召回逻辑。
4. 不扩多 hop、多塔、多阶段复杂结构。
