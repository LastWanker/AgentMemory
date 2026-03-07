# Suppressor 实现草案

## 1. 文档目标

本文档不再讨论“要不要做 suppressor”，而是直接规定：

- V5 里 suppressor 放在哪里
- 训练数据怎么来
- 模型怎么定义
- 推理时怎么接线
- 默认关闭时如何做到完全不影响主路线
- 开启时如何只轻量压掉 1 到 2 条候选
- trace / debug / 评估要输出什么

目标是让下一步编码时尽量一次到位，而不是写一半再推倒。


## 2. 对 V5 现结构的判断

先明确当前 V5 的聊天接线：

1. [retriever_adapter.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/retriever_adapter.py)
- 负责拿到 `coarse_refs`
- 负责拿到 `association_refs`
- 负责构造 `association_tags` 和 `association_trace`

2. [service.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/service.py)
- 调 `retrieve_bundle(...)`
- 把返回结果组织成 prompt
- 存会话记录
- 返回 `ChatResponse`

3. [models.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/models.py)
- `MemoryRef` 是当前候选记忆的最小交换结构

因此 suppressor 最适合挂的位置不是：

- coarse 内部
- association 内部
- DeepSeek prompt 内部

而是：

> 放在 `RetrieverAdapter.retrieve_bundle(...)` 之后，`ChatService._build_messages(...)` 之前，作为独立的候选后处理层。

这样最符合 V5 当前结构，也最容易实现“开了和没开都不影响主路线”。


## 3. 架构原则

### 3.1 必须独立

suppressor 必须是独立层，不允许改写主检索器的行为。

正确结构：

```text
coarse retriever
association retriever
    -> candidate bundle
    -> suppressor post-process
    -> final candidate bundle
    -> prompt builder
    -> LLM
```

错误结构：

```text
coarse / association 内部直接依赖 suppressor
```

### 3.2 默认 no-op

默认配置下 suppressor 应为完全关闭。

关闭时行为要求：

- 不改候选顺序
- 不改候选分数
- 不删候选
- 不影响 prompt
- 不影响 session store

也就是：

```text
input bundle == output bundle
```

### 3.3 开启后也只做轻量抑制

第一版 suppressor 的设计目标不是“大规模重排”，而是：

- 每轮最多压 1 到 2 条候选
- 只对高 suppress 分的条目动手
- 不做整包洗牌

这很重要，因为它决定了 suppressor 是“小刀”，不是“第二套检索逻辑”。


## 4. 文件与模块规划

建议新增下列模块。

### 4.1 chat 层接线模块

建议新增：

- `_V5/chat/src/chat_app/suppressor_adapter.py`

职责：

- 加载 suppressor 配置与模型
- 接收 query 与候选列表
- 返回抑制后的候选列表和 suppressor trace

### 4.2 suppressor 核心模块

建议新增：

- `_V5/src/agentmemory_v3/suppressor/__init__.py`
- `_V5/src/agentmemory_v3/suppressor/config.py`
- `_V5/src/agentmemory_v3/suppressor/model.py`
- `_V5/src/agentmemory_v3/suppressor/features.py`
- `_V5/src/agentmemory_v3/suppressor/runtime.py`
- `_V5/src/agentmemory_v3/suppressor/artifacts.py`

职责划分：

- `config.py`
  - suppressor 配置 dataclass
- `model.py`
  - plain MLP 模型定义
- `features.py`
  - query / memory 特征拼接
- `runtime.py`
  - 在线推理逻辑与惩罚逻辑
- `artifacts.py`
  - 加载模型权重、memory embeddings、id 映射、元数据

### 4.3 训练脚本

建议新增：

- `_V5/scripts/build_suppressor_dataset.py`
- `_V5/scripts/train_suppressor.py`
- `_V5/scripts/eval_suppressor.py`
- `_V5/scripts/runtime_suppressor_demo.py`

职责：

- `build_suppressor_dataset.py`
  - 从反馈、会话旧账、ANN 中构造样本
- `train_suppressor.py`
  - 训练 plain suppressor
- `eval_suppressor.py`
  - 离线评估
- `runtime_suppressor_demo.py`
  - 给定 query 和候选，查看 suppressor 会压谁


## 5. 配置设计

### 5.1 ChatAppConfig 增量字段

建议在 [config.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/config.py) 中追加：

```text
suppressor_enabled: bool
suppressor_artifact_dir: Path
suppressor_threshold: float
suppressor_alpha: float
suppressor_max_drop_per_lane: int
suppressor_keep_top_per_lane: int
suppressor_apply_to_coarse: bool
suppressor_apply_to_association: bool
suppressor_debug: bool
```

推荐默认值：

```text
SUPPRESSOR_ENABLED=0
SUPPRESSOR_THRESHOLD=0.80
SUPPRESSOR_ALPHA=0.35
SUPPRESSOR_MAX_DROP_PER_LANE=1
SUPPRESSOR_KEEP_TOP_PER_LANE=3
SUPPRESSOR_APPLY_TO_COARSE=1
SUPPRESSOR_APPLY_TO_ASSOCIATION=1
SUPPRESSOR_DEBUG=0
```

说明：

- `enabled=0` 时整套 suppressor 直接 no-op
- `threshold` 控制多高才允许压
- `alpha` 控制压的幅度
- `max_drop_per_lane` 限制每一路最多压几条
- `keep_top_per_lane` 是保险丝，防止一条路被压得太空

### 5.2 retrieval config 是否需要动

不需要。

suppressor 第一版不需要写进 V5 retrieval 主配置 YAML。

原因：

- 它不属于 coarse 或 association 的召回参数
- 它是 chat 层的后处理逻辑
- 放在 chat app config 更合理


## 6. 运行时数据结构

### 6.1 扩展 MemoryRef

建议在 [models.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/models.py) 的 `MemoryRef` 中新增字段：

```text
base_score: float = 0.0
suppressed: bool = False
suppress_score: float = 0.0
suppress_delta: float = 0.0
suppress_reason: str = ""
suppress_lane: str = ""
```

语义：

- `score`
  - 最终给前端和 LLM 的分数
- `base_score`
  - 主检索器原始分数
- `suppressed`
  - 是否被 suppressor 实际压过
- `suppress_score`
  - suppressor 模型输出的 `s`
- `suppress_delta`
  - 本轮实际扣了多少
- `suppress_reason`
  - 预留给后续更细分类别，V1 可先写 `suppressed`
- `suppress_lane`
  - `coarse` / `association`

注意：

- `base_score` 一定要保留
- 否则之后根本没法 debug“主检索其实打了多少、suppressor 扣了多少”

### 6.2 RetrievalBundle 扩展

建议在 `RetrievalBundle` 中新增：

```text
suppressor_trace: dict
```

并保持：

- `coarse_refs` 为 suppress 后的 coarse
- `association_refs` 为 suppress 后的 association

trace 单独挂出来，方便前端调试和 session 持久化。


## 7. Artifacts 设计

### 7.1 目录结构

建议 artifacts 放在：

- `data/V5/suppressor/`

建议内容：

- `manifest.json`
- `model.pt`
- `feature_spec.json`
- `memory_embeddings.npy`
- `memory_ids.json`
- `memory_texts.jsonl`
- `feedback_samples_train.jsonl`
- `feedback_samples_valid.jsonl`
- `metrics.json`

### 7.2 manifest.json

至少包含：

```json
{
  "version": "plain_mlp_v1",
  "encoder": "e5-small",
  "embedding_dim": 384,
  "feature_dim": 0,
  "model_type": "plain_mlp",
  "created_at": "",
  "train_sample_count": 0,
  "valid_sample_count": 0
}
```

后续升级 Oreo 时只需：

- `model_type: oreo_memory_mlp`

不要改整个 artifacts 体系。


## 8. 训练数据构建

### 8.1 输入来源

第一版只依赖三类数据：

1. 用户反馈
- `unrelated`
- `toforget`

2. memory embeddings / ANN

3. `_V5/chat/data/conversations`
- 提取历史 query
- 提取历史展示过的 `memory_refs`

### 8.2 build_suppressor_dataset.py 输入

建议脚本参数：

```text
--feedback-jsonl
--conversation-dir
--memory-bundle
--retrieval-config
--output-dir
--nb-size
--rand-size
--seed
```

### 8.3 unrelated 样本生成

对每条 `unrelated` 反馈：

```text
Pos:
  (q_self, m_self)

Neg_out:
  (q_self, m_nb+)

Neg_in:
  (q_nb+, m_self)

Neg_rand:
  (q_rand, m_rand)
```

其中：

- `m_nb+` 取 `NN(m_self, nb_size)`
- `q_nb+` 可来自：
  - ANN 找 query 邻居
  - 历史 query
  - 历史 memory 文本近似替代

### 8.4 toforget 样本生成

对每条 `toforget` 反馈：

```text
Pos:
  (q_forget_nb+, m_self)

Neg_named:
  (m_self, m_self)

Neg_neighbors:
  (q_forget_nb+, m_nb+)

Neg_rand:
  (q_rand, m_rand)
```

注意：

- `Neg_named` 是 V1 必须项
- 否则 `toforget` 极易学成“全局封杀”

### 8.5 输出样本格式

建议每条样本统一写成：

```json
{
  "sample_id": "",
  "split": "train",
  "feedback_type": "unrelated",
  "label": 1,
  "q_text": "",
  "m_id": "",
  "m_text": "",
  "source_kind": "pos|neg_out|neg_in|neg_named|neg_neighbors|neg_rand",
  "anchor_feedback_id": "",
  "weight": 1.0
}
```

优势：

- 训练脚本最简单
- 评估时也能按 `source_kind` 切开看


## 9. 特征方案

### 9.1 V1 主特征

V1 允许直接上 `q_vec + m_vec` 风格。

建议顺序如下：

1. 必要特征
- `feedback_type_onehot`
- `q_vec`
- `m_vec`
- `cos(q_vec, m_vec)`

2. 推荐追加
- `q_vec * m_vec`
- `abs(q_vec - m_vec)`

3. 可选轻量特征
- `explicit_mention_flag`
- `lexical_overlap_ratio`

### 9.2 暂时不做的事

V1 暂时不做：

- 使用 coarse base_score 作为主要输入
- 把 association 点亮特征塞进来
- 多通道专家模型
- 规则引擎前置

先保持特征清爽，否则第一版很难判断 suppressor 到底是样本有效还是特征凑巧。


## 10. 模型定义

### 10.1 Plain MLP V1

推荐一个保守但够用的结构：

```text
input_dim = feature_dim

Linear(input_dim, 512)
GELU
Dropout(0.15)

Linear(512, 128)
GELU
Dropout(0.10)

Linear(128, 1)
Sigmoid
```

如果输入维度太大，也可以降成：

```text
Linear(input_dim, 256)
GELU
Dropout(0.10)

Linear(256, 64)
GELU

Linear(64, 1)
Sigmoid
```

### 10.2 为什么这样就够

因为这个模型不是主语义模型，而是：

- 吃预编码向量
- 学局部抑制边界

它不需要很深，只要：

- 有一定非线性
- 容纳一点局部特例
- 不至于训练太脆


## 11. 训练流程

### 11.1 train_suppressor.py 参数

建议：

```text
--data-dir
--artifact-dir
--epochs
--batch-size
--lr
--weight-decay
--dropout
--hidden-dims
--seed
```

### 11.2 loss

第一版直接 BCE 即可：

```text
loss = BCE(pred, label, weight=sample_weight)
```

必要时允许：

- 对 `Neg_named` 稍微加权
- 对 `Neg_rand` 降低权重

但不要一开始把权重系统搞得太花。

### 11.3 验证集切分

强制按时间或 `feedback_id` 切分。

不要随机打散。

否则很容易出现：

- 同一批相近样本同时出现在 train 和 valid
- 指标虚高

### 11.4 early stopping

监控至少三类指标：

1. 总体 valid loss
2. `unrelated` 误伤率
3. 点名保留率

不能只盯 loss。


## 12. 离线评估

### 12.1 eval_suppressor.py 输出

建议输出：

- `overall_auc`
- `overall_pr_auc`
- `unrelated_pos_recall`
- `unrelated_neg_out_false_positive_rate`
- `unrelated_neg_in_false_positive_rate`
- `toforget_pos_recall`
- `toforget_neg_named_false_positive_rate`
- `toforget_neg_neighbors_false_positive_rate`
- `rand_neg_false_positive_rate`

### 12.2 最关键的产品指标

最该盯的不是总体分，而是：

1. `Neg_named` 被误压的比例
2. `Neg_in / Neg_out` 被误压的比例
3. 真 Pos 能不能明显抬高 suppress 分

如果这三件事没做好，总体 AUC 再漂亮也没意义。


## 13. 在线推理与接线

### 13.1 SuppressorAdapter 入口

建议接口：

```python
class SuppressorAdapter:
    def __init__(self, config: ChatAppConfig) -> None: ...

    def apply(
        self,
        query: str,
        lane: str,
        refs: list[MemoryRef],
    ) -> tuple[list[MemoryRef], dict]: ...
```

其中：

- `lane` 取 `coarse` 或 `association`
- 返回值：
  - 新 refs
  - trace

### 13.2 apply 的行为

逻辑固定成：

1. 如果 `SUPPRESSOR_ENABLED=0`
- 原样返回
- trace 写 `enabled=false`

2. 如果模型或 artifacts 缺失
- 原样返回
- trace 写 `status=artifact_missing`

3. 否则对每条 ref 计算 `suppress_score`

4. 按规则做轻量惩罚：
- 小于阈值：不处理
- 大于阈值：算出 `new_score = base_score * (1 - alpha * s)`

5. 仅对每一路至多压 `max_drop_per_lane` 条

6. 保证每一路至少保留 `keep_top_per_lane` 条原始高分候选不被彻底挤没

### 13.3 为什么要按 lane 分开处理

因为当前 V5 是：

- coarse 一路
- association 一路

Suppressor 不应把两路混成一锅。

否则很容易出现：

- association 本来就少，结果被 suppress 一压直接没了
- coarse 的分布把 association 完全淹死


## 14. 对 RetrieverAdapter 的改造方案

### 14.1 初始化

在 [retriever_adapter.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/retriever_adapter.py) 的 `__init__` 中新增：

```text
self._suppressor = SuppressorAdapter(config)
```

### 14.2 retrieve_bundle 改造

当前逻辑是：

1. `_retrieve_coarse`
2. `_retrieve_association`
3. 组装 `RetrievalBundle`

改成：

1. `_retrieve_coarse`
2. `_retrieve_association`
3. `apply(query, "coarse", coarse_refs)`
4. `apply(query, "association", association_refs)`
5. 组装 `RetrievalBundle`

### 14.3 suppressor_trace 的内容

建议结构：

```json
{
  "enabled": true,
  "model_type": "plain_mlp_v1",
  "lanes": {
    "coarse": {
      "candidate_count": 5,
      "applied_count": 1,
      "rows": [
        {
          "memory_id": "",
          "base_score": 0.71,
          "suppress_score": 0.86,
          "final_score": 0.49,
          "suppressed": true
        }
      ]
    },
    "association": {
      "candidate_count": 5,
      "applied_count": 0,
      "rows": []
    }
  }
}
```


## 15. 对 ChatService 的改造方案

### 15.1 prompt 不需要知道 suppressor 细节

在 [service.py](F:/GitHub/AgentMemory/_V5/chat/src/chat_app/service.py) 中：

- prompt 继续只吃最后候选
- 不需要对 LLM 额外解释 suppressor 算法

原因：

- suppressor 是后处理
- 它不是一个“新的语义来源”
- 不需要让 LLM 背负额外叙事负担

### 15.2 session store 要保留 trace

建议 assistant turn 的 metadata 中新增：

```text
"suppressor_trace": retrieval.suppressor_trace
```

这样以后离线回放时可以直接看：

- 当时压了谁
- 压了多少
- 有无误伤


## 16. runtime_suppressor_demo.py

这个脚本非常重要，因为 suppressor 比点亮算法更需要肉眼验证。

建议功能：

```text
python _V5/scripts/runtime_suppressor_demo.py --query "xxx" --top-k 5
```

输出：

1. coarse 原始候选
2. coarse suppress 后候选
3. association 原始候选
4. association suppress 后候选
5. 每条的：
   - base_score
   - suppress_score
   - final_score
   - 是否被压

没有这个脚本，后面调 suppressor 会很痛苦。


## 17. V1 行为约束

为了避免 suppressor 过早失控，第一版强制加以下约束：

1. 默认关闭
2. 只做后处理
3. 每一路最多压 1 条
4. `threshold` 先设高一些
5. `alpha` 先保守
6. 不允许 suppressor 让整个 lane 清空
7. 不允许 suppressor 参与候选生成

这几条不是“保守主义”，而是为了保持它仍然是 V5 的一个可拔插层。


## 18. 未来升级到 Oreo 的接口预留

第一版虽然只实现 plain MLP，但接口要为 Oreo 留钩子。

### 18.1 manifest 驱动模型类型

`manifest.json` 中保留：

```text
model_type = plain_mlp_v1 | oreo_memory_mlp_v1
```

### 18.2 runtime 按 model_type 分派

`runtime.py` 里做：

```python
if manifest.model_type == "plain_mlp_v1":
    ...
elif manifest.model_type == "oreo_memory_mlp_v1":
    ...
```

不要把 plain 和 Oreo 的逻辑写死在两个完全不同的入口里。

### 18.3 特征管道保持兼容

Plain 与 Oreo 尽量共享：

- 样本 jsonl
- feature builder
- artifacts manifest
- eval 流水线

这样之后从 plain 升级 Oreo 才不会又重建一遍系统。


## 19. 第一轮实现顺序

建议直接按这个顺序做，不要跳步。

### 第 1 步

补 chat config 与 models：

- `ChatAppConfig` 增加 suppressor 配置
- `MemoryRef` 增加 suppress 字段
- `RetrievalBundle` 增加 `suppressor_trace`

### 第 2 步

实现 artifacts / feature / plain model：

- `artifacts.py`
- `features.py`
- `model.py`

### 第 3 步

实现 `SuppressorAdapter.apply(...)`：

- 能加载 artifact
- 能对候选计算 suppress 分
- 能做轻量惩罚
- 能输出 trace

### 第 4 步

把 suppressor 接到 `RetrieverAdapter.retrieve_bundle(...)`

### 第 5 步

写 `runtime_suppressor_demo.py`

### 第 6 步

写 `build_suppressor_dataset.py / train_suppressor.py / eval_suppressor.py`

注意：

- 先把 runtime 接口和 no-op 行为打通
- 再补训练闭环

这样即使还没正式训练模型，也可以先用假 artifact / mock 分数把链路跑通。


## 20. 一些明确不做的事

第一版明确不做：

1. suppressor 直接改召回器内部逻辑
2. suppressor 参与 query 改写
3. suppressor 和 association 点亮图深度耦合
4. 复杂规则引擎
5. 多模型 ensemble
6. 训练时就混入太多主检索信号

这些都会把 suppressor 从“可拔插小层”变成“第二主系统”，这不符合当前目标。


## 21. 完成标准

如果实现后满足下面这些条件，就算 V1 成了：

1. `SUPPRESSOR_ENABLED=0` 时，系统行为与现在完全一致
2. `SUPPRESSOR_ENABLED=1` 时，coarse / association 各自最多只会被轻压 1 条
3. 前端和 session trace 能看到 suppressor 的分数与惩罚结果
4. 不需要修改 LLM prompt 主体逻辑
5. artifacts 缺失时会优雅退回 no-op
6. 后续切 Oreo 时不需要重写 chat 接线层


## 22. 最终拍板

Suppressor 在 V5 里的正确身份是：

- 独立层
- 后处理层
- 默认关闭
- 开启后只轻压少数候选
- 不改变主检索路线

如果这条身份线守住了，那么 suppressor 将是一个非常适合 V5 的增强件：

- 开关干净
- 风险可控
- 收益明确
- 也为后面的 Oreo 容量升级留下了完整接口

