# Suppressor 与 Oreo 全量提纲

## 1. 文档定位

本文档用于替代旧的 `suppressor.txt`，把当前已经讨论过的 suppressor 主方案，以及后续可能升级的 Oreo 夹心方案，一次性写清楚。

目标不是做一个“新的主检索器”，而是在现有粗召回、联想召回之外，增加一个**局部、尖锐、偏保守**的抑制模块：

- 它不负责发现相关内容
- 它不负责理解用户全局意图
- 它不替代 coarse / association
- 它只负责一件事：在某些 query 邻域里，把某些明确令人讨厌、明显不该再出来的 memory 压下去

一句话概括：

> Suppressor 不是 reranker，而是一个只做减法的局部抑制器。


## 2. 要解决的核心问题

当前主检索链已经能把“相关的东西”大致找回来，但用户反馈里仍然会出现两种非常棘手的负反馈：

1. `unrelated`
- 用户当前这句话下，这条 memory 明显不相关
- 但这条 memory 可能并不是全局无用，只是在这一类问法下不该出现

2. `toforget`
- 用户希望某条 memory 不再被带出来，或者至少明显降低出现概率
- 但这不等于“无论什么 query 都不准再出现”
- 尤其在“明确点名”场景下，更不能把它全局封杀

所以 suppressor 的本质任务不是“判断相关性”，而是：

- 学会“哪一类问法下，要压哪一类 memory”
- 同时避免：
  - 把相邻 query 一起误伤
  - 把相邻 memory 一起误伤
  - 把某条 memory 全局打死


## 3. 总体原则

### 3.1 角色边界

Suppressor 的角色边界必须非常明确：

- 输入：`(query, memory)` 对
- 输出：`s(q, m) in [0, 1]`
- 语义：这对组合应不应该被压，以及压多少

它不是：

- 全局排序器
- 新的主召回器
- 新的端到端问答模型
- 一个复杂规则引擎

### 3.2 上线方式

上线时 suppressor 只做扣分，不做加分。

推荐第一版口径：

1. 先算出主检索链自己的 `base_score`
2. suppressor 产出 `s(q, m)`
3. 若 `s` 低于阈值，则不处理
4. 若 `s` 高于阈值，则施加乘法惩罚

可用形式：

```text
if s <= threshold:
    final_score = base_score
else:
    final_score = base_score * (1 - alpha * s)
```

保留的备用形式：

```text
final_score = base_score - lambda * s
```

当前更偏向前者，因为它更像“抑制系数”，也更容易讲故事。

### 3.3 训练目标

训练目标不是学通用相关性，而是学“局部抑制边界”：

- 该压的时候要尖锐
- 不该压的时候尽量别误伤
- 它可以带有一点主观偏好
- 它不要求像通用语义匹配模型那样强泛化

换句话说：

- 一点点过拟合并不是灾难
- 只要误伤控制得住，模型尖一点反而可能更符合产品目标


## 4. 数据与索引

### 4.1 基础反馈记录

最小反馈单元：

- `feedback_id`
- `ts`
- `type: enum {unrelated, toforget}`
- `q_text`
- `m_id`
- `m_text_snapshot` 可选，用于可追溯

### 4.2 MemoryIndex

Suppressor 训练阶段需要可查询的记忆索引：

- `m_id -> m_vec`
- ANN 索引：用于 `NN(m, k)` 和 `NN(q, k)` 的近邻检索

### 4.3 会话旧账

当前阶段不强依赖正式 CandidateCache。

可直接离线翻：

- `_V5/chat/data/conversations`

里面已有历史对话和当时展示过的 `memory_refs`，可以视为“用户看到但没有正反馈/特殊反馈的候选池旧账”。

当前阶段建议：

- 先靠历史会话旧账 + ANN 跑通 suppressor
- 等 suppressor 本身确实有价值，再考虑包装成正式数据库 / CandidateCache

### 4.4 CandidateCache 的地位

CandidateCache 未来可以补，但现在不是阻塞项。

形式可设想为：

- `feedback_id -> candidate_m_ids`

它有价值，但当前阶段不应为了它去拖慢 suppressor 主体推进。


## 5. 任务定义

统一任务形式：

```text
Suppressor(q, m, feedback_type) -> s in [0, 1]
```

其中：

- `q` 是 query 文本或其向量表示
- `m` 是 memory 文本或其向量表示
- `feedback_type` 必须显式输入

这里强制保留 `feedback_type` 的原因：

- `unrelated` 和 `toforget` 虽然都叫“抑制”
- 但语义并不完全一样
- 用户希望模型在一个统一框架内学到两种不同的抑制偏好
- 因此先做**单模型 + feedback_type 特征**，而不是拆成两个模型

这也是当前叙事上最顺的做法：

- 一个 suppressor
- 两种反馈语境
- 不同语境下学不同的抑制偏好


## 6. 样本构造总原则

Suppressor 的样本不是普通二分类样本，而是带强产品语义的 Pos/Neg 设计。

核心思想：

- Pos 表示“这对 `(q, m)` 应该被压”
- Neg 表示“这对 `(q, m)` 不应该被压”

也就是说：

- `s` 高：应抑制
- `s` 低：不应抑制

两个关键原则：

1. 必须有“局部性”
- 只压这个 query 邻域里的这个 memory

2. 必须有“双保护面”
- 不把 query 邻域整体压坏
- 不把 memory 邻域整体压坏


## 7. unrelated 样本构造

### 7.1 目标语义

`unrelated` 表示：

- 这条 memory 在当前 query 下不相关
- 但并不意味着这条 memory 对所有 query 都不相关

### 7.2 Pos

```text
Pos: (q_unrel_self, m_unrel_self)
```

表示：

- 这就是用户明确点了“不相关”的那对 `(q, m)`
- 模型应该学会把它压下去

### 7.3 Neg_out

```text
Neg_out: (q_unrel_self, m_unrel_nb+)
```

语义：

- 同一个 query 下，别的相邻 memory 不应该被顺手压掉

否则模型很容易学成：

- “只要看到这句 query，就整片 memory 都压”

### 7.4 Neg_in

```text
Neg_in: (q_unrel_nb+, m_unrel_self)
```

语义：

- 同一条被标不相关的 memory，在相邻 query 下不一定也该被压

否则模型会学成：

- “这条 memory 天生有罪”

### 7.5 Neg_rand

```text
Neg_rand: (q_rand, m_rand)
```

这里 `rand-rand` 的定位不是 hard negative，而是：

- 背景校准器
- 黑盒抚平器
- 防止模型内部通道变形后，把完全不相干组合也打出奇怪高分

所以它不是没用，而是：

- 有用
- 但不该成为主力样本

### 7.6 邻域来源

`q_unrel_nb` 的来源：

- 可用 `NN(q_unrel_self)` 从历史 query 或 memory 文本中近似得到
- 即便名字叫 `q_unrel_nb`，来源也允许是 memory 文本或历史会话文本

`m_unrel_nb` 的来源：

- `NN(m_unrel_self, nb_size)`

### 7.7 为什么 unrelated 的双保护面是刚需

`(q_self, m_nb)` 与 `(q_nb, m_self)` 两类 Neg 缺一不可。

少一类都会出问题：

- 少前者：模型会把这个 query 邻域整体打黑
- 少后者：模型会把这条 memory 全局打黑


## 8. toforget 样本构造

### 8.1 目标语义

`toforget` 表示：

- 用户不希望某条 memory 再那么容易被带出来
- 但这依然不是“所有 query 下永久封杀”

### 8.2 Pos

```text
Pos: (q_forget_nb+, m_forget_self)
```

语义：

- 不一定非得用原始同句 query
- 更重要的是让模型理解“与这条 memory 相邻的一类问法下，要对这条 memory 提高抑制”

### 8.3 Neg_named

```text
Neg_named: (m_forget_self, m_forget_self)
```

这是 `toforget` 里非常关键的一类 Neg。

它的产品语义是：

- 如果 query 明确点名该 memory 本身
- suppressor 不应继续硬拦

这就是“点名时别拦”的核心近似。

### 8.4 Neg_neighbors

```text
Neg_neighbors: (q_forget_nb+, m_forget_nb+)
```

语义：

- 不要因为用户想压一条 memory
- 就把它附近的其他 memory 也一起压了

### 8.5 Neg_rand

```text
Neg_rand: (q_rand, m_rand)
```

这里依旧作为背景校准器使用。

### 8.6 邻域来源

`q_forget_nb` 的来源：

- `NN(q_forget_self)` 或相关替代 query 邻域

`m_forget_nb` 的来源：

- `NN(m_forget_self, nb_size)`，且不含自身

### 8.7 toforget 的关键点

`Neg_named` 必须存在。

如果没有这类 Neg，模型很容易滑向：

- “这条 memory 永远别出现”

这和 suppressor 的设计初衷相违背。


## 9. 样本比例与批次建议

建议把样本分成三类看：

1. 主 Pos
2. hard negatives
3. rand-rand 背景 negatives

推荐原则：

- hard negatives 占大头
- rand-rand 保留，但占比明显低于 hard negatives
- 不要让 rand-rand 淹没训练信号

建议的理解方式：

- hard negatives 决定边界长什么样
- rand-rand 负责抚平奇怪通道

如果 `rand-rand` 太多，模型可能学成：

- 对大多数东西都打低 suppress 分
- 结果看似稳，实际没学到有效抑制能力


## 10. V1 主方案：Plain Suppressor

### 10.1 结构定位

第一版先做 plain MLP baseline。

它的意义不是最终答案，而是：

- 先验证 suppressor 这个机制本身值不值得做
- 先验证样本构造是否真的能得到“尖锐但不误伤”的效果
- 先建立评估基线

### 10.2 为什么允许直接喂 `q_vec + m_vec`

当前口径接受：

- 直接把 `q_vec + m_vec` 喂给小 MLP 作为 V1

原因：

- 这个任务不追求教科书式完美泛化
- 它本来就允许一点主观偏好
- 适度过拟合不一定是坏事
- 只要误伤控制住，模型更尖锐可能反而更符合需求

### 10.3 推荐输入特征

V1 可接受的输入特征：

- `feedback_type` 必须有
- `q_vec`
- `m_vec`
- `cos(q_vec, m_vec)`
- `q_vec ⊙ m_vec` 可选
- `|q_vec - m_vec|` 可选
- `lexical_overlap` 可选
- `explicit_mention_flag` 可选，但强烈建议保留

当前不建议一开始把太多主检索器内部信号塞进来，尤其是：

- 暂时不把 `base_score` 当成主要模型输入

这样做的好处是：

- suppressor 与主检索器耦合更弱
- 后续 coarse / association 打分变动时，suppressor 不至于跟着一起漂

### 10.4 一个合理的 V1 网络形状

可以用：

```text
[input]
  -> Linear
  -> GELU / ReLU
  -> Dropout
  -> Linear
  -> GELU / ReLU
  -> Dropout
  -> Linear
  -> Sigmoid
```

“三层 MLP 很难真的过拟合”这件事，在这里大体成立，但仍建议保守些：

- hidden size 不要过大
- 先小模型
- 先看训练曲线和误伤率

### 10.5 防过拟合手段

虽然这个任务容忍一点过拟合，但还是建议保留便宜的防过拟合手段：

- 小 hidden size
- dropout
- weight decay
- early stopping
- 验证集按时间或 `feedback_id` 切分，不要乱混


## 11. V1 上线逻辑

Suppressor 不改召回，只改候选分数。

推荐接法：

1. coarse / association 先各自召回
2. 得到候选 `(q, m, base_score)`
3. 对每个候选跑 suppressor
4. 当 `s > threshold` 时施加惩罚
5. 再把惩罚后的结果送入最终展示/LLM 输入

当前更推荐：

- suppressor 先只作为后处理
- 不嵌入召回阶段
- 不参与候选生成

这样便于：

- debug
- ablation
- shadow mode 验证


## 12. 评估指标

Suppressor 的评估不能只看一个 AUC。

必须按产品语义拆开看：

### 12.1 unrelated 方向

- `unrelated` 的抑制命中率
- `unrelated` 的邻居误伤率
- 同 query 邻域下，其他候选被错压的比例

### 12.2 toforget 方向

- `toforget` 的抑制命中率
- 相邻 memory 被误伤的比例
- 该 memory 在无关 query 邻域下被过度抑制的比例

### 12.3 点名保护

- 明确点名 query 下的召回保留率

这一项尤其重要，因为 `toforget` 最容易把“点名查询”也一起打死。

### 12.4 线上感知指标

- 用户再次点到相同负反馈的频率是否下降
- 看起来“莫名其妙”的误召回是否下降
- suppressor 是否把系统整体变钝


## 13. 数据来源的现实版本

当前阶段不额外发明豪华数据系统。

先用：

- 用户已有负反馈
- `_V5/chat/data/conversations` 里的历史 `memory_refs`
- ANN 近邻

先把 suppressor 的核心能力验证出来：

- 能否压住该压的 pair
- 是否避免明显误伤

等这个核心价值成立，再去做：

- 正式 CandidateCache
- 数据库存储
- 更干净的训练流水线


## 14. 为什么暂时不拆成两个模型

虽然 `unrelated` 和 `toforget` 语义不同，但当前不拆双模型。

原因：

1. 单模型更简洁
2. `feedback_type` 已经能告诉模型当前语境
3. 用户希望模型能在统一参数里学到某些“偏好”
4. 单模型更利于讲故事和产品表达

所以当前方案明确为：

- 一个 suppressor
- 一个共享主干
- 一个 `feedback_type` 特征

如果后面发现：

- 两类反馈强烈互相干扰
- 一类明显拖累另一类

再考虑分裂也不迟。


## 15. Oreo 方案的定位

### 15.1 Oreo 不是推翻，而是升级

Oreo 夹心方案不是另起炉灶，而是 plain suppressor 的容量升级版。

它适用于这样的场景：

- plain MLP 已经能学会一般抑制边界
- 但遇到很尖锐、很条件化、很像“查表特例”的 pair 时，容量不够精细
- 或者 plain MLP 为了记住少量特例，把全局参数扭歪了

所以 Oreo 更像：

- V1.5
- 或 V2

而不是第一枪必须上的东西。

### 15.2 Oreo 的基本直觉

有些 suppressor 模式不是平滑规则，而像局部禁忌：

- 只有某类 query 邻域下
- 某条 memory 才需要被强力压制

这种东西天然有“查表味”。

如果全部靠普通 MLP 参数硬记，会有两个风险：

1. 为了记住少量特例，把全局参数拧坏
2. 某些词或局部模式被学成“大棒”，导致大面积误伤

Oreo 的思路是：

- 前两层 MLP 先把 `(q, m)` 压成中间表示 `z`
- 用 `z` 去读一张稀疏激活的 key-value 记忆表
- 让模型把那些很尖锐的特例记到少数槽位里
- 再把读出来的记忆向量 `mem` 与 `z` 一起送进后两层 MLP 输出抑制分数


## 16. Oreo 的结构定义

### 16.1 总体结构

```text
input features
  -> MLP block A (2层)
  -> latent z
  -> memory layer (key-value, sparse read)
  -> mem
  -> concat(z, mem)
  -> MLP block B (2层)
  -> sigmoid
  -> s(q, m)
```

### 16.2 输入特征

推荐从较克制的特征开始：

- `feedback_type`
- `cos(q_vec, m_vec)`
- `q_vec ⊙ m_vec`
- `|q_vec - m_vec|`
- 需要时再加：
  - `q_vec`
  - `m_vec`
  - 少量标量特征

### 16.3 memory layer

memory layer 由 `K` 个槽位组成：

- 每个槽位有一个 `key`
- 每个槽位有一个 `value`

读取过程：

1. 用 `z` 与所有 `key` 算相似度
2. 用低温 softmax 或相近机制得到槽位权重
3. 只保留 top-r 个槽位
4. 其他槽位权重直接截断为 0
5. 对对应 `value` 加权求和，得到 `mem`

### 16.4 为什么要 top-r

`top-r` 是 Oreo 成立的关键。

如果不稀疏读取，memory layer 很容易退化成：

- 又一层普通连续变换

而 suppressor 真正需要的是：

- 局部记忆
- 少数槽位负责少数尖锐模式

推荐初值：

- `r = 2`

### 16.5 为什么它像“可学习查表”

因为它允许模型做这样的事：

- 一些很尖锐的特例 `(q, m)` 会稳定命中同一个槽位
- 槽位记住的是“某种失败模式 / 禁配模式”
- 相邻 query 可能共享这个槽位，产生有限泛化
- 不相干 query 基本不会激活这个槽位

这样就能避免最怕的情况：

- 某个局部词模式被学成全局大棒


## 17. Oreo 的训练方式

Oreo 训练时不改原有 Pos/Neg 采样体系。

仍然沿用：

- unrelated 的 Pos / 双保护面 Neg / rand-rand
- toforget 的 Pos / `Neg_named` / `Neg_neighbors` / rand-rand

也就是说：

- 训练规则不变
- 改的是模型容量与表达方式

这是 Oreo 工程上最大的好处之一：

- 不需要推翻样本工程
- 不需要新增复杂在线依赖
- 不需要拆成一堆显式模块


## 18. Oreo 的正则与训练约束

Oreo 的风险不是“原理错”，而是“训练容易太激进”。

因此建议从最小版开始。

### 18.1 先上的东西

- top-r 截断
- 较小的温度
- 监控槽位使用分布

### 18.2 暂时不要一开始全上

以下东西先作为可选项，不要首轮全堆：

- 很强的 usage balance regularizer
- 很强的熵惩罚
- 很大的槽位数
- 很尖到近乎硬路由的温度

### 18.3 推荐的最小 Oreo

可从下面这套开始：

- `K = 16` 或 `32`
- `r = 2`
- `MLP(2层) -> memory -> MLP(2层)`
- 先只监控槽位使用情况
- 发现明显塌缩后，再轻量加约束


## 19. 什么时候值得从 Plain 升级到 Oreo

当出现以下症状时，说明 Oreo 值得试：

1. plain MLP 记不住很尖锐的特例
2. 为了记住少量特例，误伤面积明显变大
3. `unrelated` 与 `toforget` 的边界老是缠在一起
4. 某些特定反馈模式很像局部“禁配表”

如果还没出现这些症状，就没必要急着上 Oreo。


## 20. Plain 与 Oreo 的关系

两者不是二选一，而是前后版本关系。

### 20.1 Plain 的价值

- 快
- 简单
- 好训
- 好验
- 能先验证样本设计是否站得住

### 20.2 Oreo 的价值

- 增加“局部记忆”能力
- 对稀疏特例更友好
- 更适合 suppressor 这种有“禁忌表”气质的任务

### 20.3 当前建议

建议推进顺序：

1. 先做 plain suppressor baseline
2. 看它的误伤与尖锐性是否平衡
3. 若 plain 的容量明显不够，再升级到 Oreo


## 21. 推荐实施顺序

### 阶段 A：先把 suppressor 跑通

1. 整理负反馈样本
2. 接 MemoryIndex / ANN
3. 从历史会话旧账里回收候选上下文
4. 按 `unrelated / toforget` 规则构造 Pos / Neg
5. 训练 plain MLP baseline
6. 做离线评估与 shadow mode

### 阶段 B：上线后处理抑制

1. 对 coarse / association 候选逐条打 suppress 分
2. 使用阈值 + 乘法惩罚
3. 观察误伤与收益

### 阶段 C：若 plain 不够，再做 Oreo

1. 保持样本规则不变
2. 将 MLP 升级为 `MLP -> memory -> MLP`
3. 先上小槽位、低复杂度版本
4. 比较是否在不增加误伤的情况下提升尖锐抑制能力


## 22. 当前冻结口径

当前讨论后，冻结如下：

1. suppressor 是局部抑制器，不是 reranker
2. 单模型，不拆成 unrelated / toforget 两个模型
3. `feedback_type` 必须显式输入
4. V1 允许直接用 `q_vec + m_vec` 喂小 MLP
5. `rand-rand` 保留，定位为背景校准器，不是主力 hard negative
6. 当前先不强依赖 CandidateCache，先用 `_V5/chat/data/conversations` 的旧账 + ANN
7. 当前先不把 `base_score` 当主要模型输入
8. 上线先走阈值 + 乘法惩罚
9. Oreo 是升级版，不是第一版刚需
10. Oreo 的核心价值是“可学习查表式的局部记忆”


## 23. 最后的判断

这个方向之所以值得做，不在于它多先进，而在于它非常贴合当前项目：

- coarse 负责稳
- association 负责活
- suppressor 负责别把某些讨厌的东西老推上来

它是一个明显偏工程、偏产品、偏局部控制的模块。

而 Oreo 的意义，在于如果 plain suppressor 已经证明这个方向成立，那么可以进一步把“尖锐性”和“局部性”同时做强，而不必把整个系统搞成规则引擎。

所以最健康的路线不是空谈“终极方案”，而是：

1. 先把 plain suppressor 做成一个能验、能看、能上线影子跑的版本
2. 再决定 Oreo 是否值得作为容量升级

