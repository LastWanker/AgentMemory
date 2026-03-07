# data/V5 suppressor 资产处置建议（按标题判断）

更新时间：2026-03-06  
判定口径：**只看目录/文件标题，不看内容质量**。

## 1. 结论先行

为记忆网络方案转向，建议：

1. **可复用仅保留一版**：`data/V5/suppressor_oreo_r1_dataset`
2. 其余 `suppressor*` 目录全部视为旧路线资产，可清理。

## 2. 建议保留（仅 1 版）

1. `data/V5/suppressor_oreo_r1_dataset`
   - 标题显示它是“dataset”，且包含 `feedback_groups_*` 与 `feedback_samples_*`
   - 对记忆网络路线可作为首版训练/回放种子数据

建议后续重命名为：

`data/V5/mem_network_suppressor_seed_dataset`

## 3. 建议直接删除（旧 suppressor 路线）

1. `data/V5/suppressor`
2. `data/V5/suppressor_calib`
3. `data/V5/suppressor_calib_mix`
4. `data/V5/suppressor_newfb`
5. `data/V5/suppressor_newfb_v2`
6. `data/V5/suppressor_newfb_v3`
7. `data/V5/suppressor_newfb_v4`
8. `data/V5/suppressor_oreo_r1_artifact`
9. `data/V5/suppressor_oreo_r1_eval`
10. `data/V5/suppressor_oreo_r2_eval`
11. `data/V5/suppressor_oreo_r2_typeheads_artifact`
12. `data/V5/suppressor_oreo_r2_typeheads_artifact_v2`
13. `data/V5/suppressor_realcheck`

## 4. 处置顺序（降低风险）

1. 先把 `suppressor_oreo_r1_dataset` 复制为新名字（mem_network seed）
2. 再删除上述旧目录
3. 删除后做一次目录核对，只剩 1 版 seed

## 5. 备注

这个建议严格按“标题语义”做，不依赖内容细看。  
如果你同意，我下一步可以直接执行“复制一版 seed + 批量删除旧目录”。
