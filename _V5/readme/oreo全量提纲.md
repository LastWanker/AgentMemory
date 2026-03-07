# Oreo 全量提纲（进取版：Oreo + Pairwise/Listwise 一体化）

更新时间：2026-03-05

## 1. 文档目的

本文档用于冻结 V5 suppressor 下一阶段的设计口径：

1. 不再只做 plain MLP 二分类抑制
2. 正式引入 Oreo 夹心层（memory layer）
3. 训练目标升级为 query 内相对排序（pairwise/listwise）
4. 保持现有 chat 接线与主检索链稳定，不破坏 coarse/association 主功能

目标是解决当前已验证的二难：

- 参数激进：feedback 压制有效，但随机扰动过高
- 参数保守：随机扰动很低，但 feedback 压制弱

## 2. 保留不变的原则（继续有效）

以下原则沿用，不改：

1. suppressor 是后处理层，不侵入 coarse / association 召回器内部
2. 默认 no-op，关掉时主链行为不变
3. coarse 与 association 分 lane 独立抑制
4. 每 lane 有硬保险丝：`max_drop_per_lane`、`keep_top_per_lane`
5. 评测必须双 cohort（feedback cohort + random cohort）

## 3. 现状问题复盘

当前 plain 方案本质是 pointwise 打分 + 门控扣分：

- 模型给 `s(q,m)`，再由阈值规则决定是否压制
- 训练时主要学习“这对 pair 是否该压”

问题在于：

1. 目标和线上真实决策不一致
- 线上是在“同 query 候选集内”决定压谁
- 训练却是“独立 pair 二分类”

2. 容量不足以表达局部禁忌模式
- 某些反馈呈现“查表式”局部模式
- plain MLP 容易在“记住特例”和“全局误伤”之间摆动

因此需要“结构升级 + 目标升级”同步进行。

## 4. 进取版核心方案

核心是双升级：

1. 模型：`Plain -> Oreo`
2. 目标：`Pointwise -> Query-local Pairwise/Listwise`

### 4.1 模型升级（Oreo）

结构：

```text
features(q,m,type,lane,base_context)
  -> Block A (2-layer MLP)
  -> latent z
  -> Memory Layer (K slots, top-r sparse read)
  -> mem(z)
  -> concat(z, mem, small scalar features)
  -> Block B (2-layer MLP)
  -> sigmoid
  -> suppress score s(q,m)
```

memory layer 参数：

- `K`: 槽位数（建议 16/32 起步）
- `r`: top-r 稀疏读取（建议 2）
- `tau`: 温度（可调，避免塌缩）

### 4.2 目标升级（Pairwise/Listwise）

每个 query 形成候选组 `Cq`，在组内学“该压谁，不该压谁”。

训练目标建议组合：

1. Pairwise 主损失（必选）
- 对 `(m_pos, m_neg)` 约束 `s(q,m_pos) > s(q,m_neg)`

2. Listwise 辅损失（推荐）
- 在 `Cq` 上对 suppress 分布做归一化监督
- 强化“同 query 内排序形状”

3. Pointwise BCE（弱权重保留）
- 作为稳定器，不再是主目标

## 5. 数据与样本升级（V2）

## 5.1 样本单位

从“独立 pair”升级为“query 组”：

```text
group_id (query-level)
  q_text
  feedback_type
  lane
  candidates: [m1, m2, ...]
  labels: should_suppress / should_keep
```

## 5.2 候选来源

按当前线上真实路径对齐：

1. 反馈事件中的 `effective_candidate_refs`
2. 若不足，补充 ANN 邻域
3. 保留随机背景样本，但不主导比例

## 5.3 标签语义

- `unrelated`:
  - 正例：被点的 `(q_self, m_self)`
  - 保护面：`(q_self, m_nb)`、`(q_nb, m_self)` 必须保留
- `toforget`:
  - 正例：forget 邻域下的 `(q_nb, m_self)`
  - 强保护：`Neg_named=(m_self,m_self)` 必须保留

## 5.4 全局偏置去除（新增）

引入 memory 级先验偏置（bias）用于抚平“大棒效应”：

- 统计 `memory_id` 在全体样本中的平均 suppress 倾向
- 线上推理时做校正：`s_calibrated = s_raw - lambda_bias * bias(memory_id)`
- 对低频 memory 做平滑，避免噪声放大

这一步是“非上下文门控”的低成本抑噪核心。

## 6. 特征口径（Oreo 版）

维持简洁，避免重新耦合主检索器：

1. 必选
- `feedback_type_onehot`
- `lane_onehot`
- `cos(q,m)`
- `q*m`
- `|q-m|`
- `lexical_overlap`
- `explicit_mention`

2. 可选
- `q_vec`、`m_vec`（开关控制）
- `base_score`（先弱特征，不作为主导）

推荐默认：

- 先用 `product + abs_diff + cosine + scalar` 主干
- `q_vec/m_vec` 作为可切换 ablation

## 7. 在线推理与门控（保持兼容）

在线仍使用当前 runtime 门控框架，但语义升级为“排序后处理”：

1. 先算每条候选 `s_calibrated`
2. 计算 `predicted_delta = base_score * alpha * s_calibrated`
3. 过门控：
- threshold
- confidence_margin
- min_delta
- min_zscore
4. 每 lane 仅选 top `max_drop_per_lane` 执行
5. `applied_count==0` 时禁止重排（继续保留）

新增建议：

- `min_delta` 支持按 lane 分别配置（coarse/association 分布差异明显）

## 8. Artifacts 规范（新增 Oreo 字段）

目录仍沿用 `data/V5/suppressor_*`，但 manifest 增加：

```json
{
  "model_type": "oreo_memory_mlp_v1",
  "objective": "pairwise_listwise",
  "memory_layer": {
    "slots_k": 32,
    "top_r": 2,
    "tau": 0.7
  },
  "feature_flags": {...},
  "bias_calibration": {
    "enabled": true,
    "lambda_bias": 0.2
  }
}
```

建议产物：

1. `model.pt`
2. `manifest.json`
3. `feature_spec.json`
4. `memory_embeddings.npy`
5. `memory_ids.json`
6. `memory_texts.jsonl`
7. `memory_bias.json`（新增）
8. `metrics.json`

## 9. 脚本改造规划

## 9.1 训练脚本

在现有 [train_suppressor.py](F:/GitHub/AgentMemory/_V5/scripts/train_suppressor.py) 增强：

新增参数：

- `--model-type plain|oreo`（默认 oreo）
- `--objective pointwise|pairwise|listwise|hybrid`（默认 hybrid）
- `--slots-k`
- `--top-r`
- `--tau`
- `--lambda-pairwise`
- `--lambda-listwise`
- `--lambda-pointwise`
- `--enable-memory-bias`
- `--lambda-bias`

## 9.2 数据脚本

在现有 [build_suppressor_dataset.py](F:/GitHub/AgentMemory/_V5/scripts/build_suppressor_dataset.py) 增强：

- 输出 query-group 训练文件
- 保留旧 pair 文件用于回退

新增输出建议：

- `feedback_groups_train.jsonl`
- `feedback_groups_valid.jsonl`

## 9.3 评测脚本

保留 [tune_suppressor_two_cohorts.py](F:/GitHub/AgentMemory/_V5/scripts/tune_suppressor_two_cohorts.py) 主入口，新增两类指标：

1. feedback 定向压制指标
- “被标记 memory 在 top5 消失率”
- “被标记 memory 排名后移幅度”

2. random 稳定性指标
- top1 改变率
- top5 集合改变率
- top5 顺序改变率

## 10. 验收标准（本阶段）

以双 cohort 为准：

1. A 组（feedback query）
- `feedback_pair_removed_rate` 显著提升（相对 plain 基线）

2. B 组（random 2000）
- `random_dual_top1_changed_rate` 与 `random_dual_top5_set_changed_rate` 维持可控区间

3. 回归要求
- `SUPPRESSOR_ENABLED=0` 与当前线上行为完全一致
- artifact 缺失时 graceful no-op

## 11. 分阶段实施顺序

1. 先补模型与 artifact 分派
- 在 runtime 中按 `model_type` 分支 plain/oreo

2. 再补训练目标
- 先上 pairwise
- 再加 listwise

3. 再加 bias 校正
- 先离线启用
- 验证后接在线

4. 最后统一调参与双 cohort 验收

## 12. 风险与控制

风险：

1. Oreo 训练不稳定（槽位塌缩）
2. pairwise/listwise 目标导致过拟合反馈集
3. bias 校正过强导致 suppress 无效化

控制：

1. 小 K 起步（16/32）+ top-r=2
2. 先 hybrid loss，pointwise 不完全移除
3. bias 加平滑与上限裁剪
4. 每次改动必须回跑双 cohort

## 13. 最终结论

当前阶段继续只调 plain 参数，收益已经接近上限。  
要突破“feedback 与 random 扰动的二难”，必须同步升级：

1. 结构容量（Oreo 夹心）
2. 训练目标（query 内相对排序）

因此本提纲拍板为：

- **V5 suppressor 下一阶段默认路线 = Oreo + Pairwise/Listwise 一体化**
- 保留 plain 作为可回退基线，不再作为主推进方向
