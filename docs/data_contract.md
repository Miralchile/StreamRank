# 数据契约

## Interaction

| 字段 | 语义 |
|---|---|
| `user_id` | 数据集版本内部重映射 ID,不跨 Pure/1K/27K 复用 |
| `item_id` | `video_id` 的内部统一命名 |
| `event_time_ms` | 原始 `time_ms`,事件发生时间 |
| `logging_policy` | `standard` 或 `random`(由 `is_rand` 推断),评测时分开聚合 |
| `tab` | 场景字段,不自动映射 UI |
| `category` / `author_id` / `upload_time_ms` | 来自 `video_features_basic_pure.csv` 的物品元数据 |

## 标签

- `is_click`:UI 相关的 click/valid-play 复合语义。
- `long_view`:播放时长阈值标签。
- `is_like`:点赞,辅助任务。
- `is_hate`:明确负反馈,稀疏风险诊断。

标签角色(主/辅助/诊断)在 README 与评测协议中固定,不随结果好坏事后调整。

## 时间正确性规则

任意训练/评测样本的特征(用户历史序列、用户与物品统计)只能来自事件时间更早的时间桶:

```text
source_event.bucket < x_t.bucket    (bucket = event_time_ms // 1000ms)
```

同一时间桶近似同一请求:桶内所有曝光先生成样本、后消费反馈。serving engine 的 artifact
有明确训练截止时间;早于该截止时间的查询会被直接拒绝,不对预载全量索引做回溯查询。
候选宇宙只包含 `max(upload_time, first_seen) <= query_time` 的物品。

## 禁用/受限字段

- `video_features_statistic.csv`:默认禁用,因其为跨月聚合而非逐时刻快照。
- `visible_status`:只能做当前快照敏感性分析,不能复原历史可用性。
- `upload_dt`:只允许 `upload_time <= query_time` 的物品进入候选宇宙。
- author/music ID:冷启动时需要 OOV;不能假设它们总在训练期出现。

## 负样本

- 排序:真实曝光但目标行为为 0;不引入未曝光负采样。
- 新模型召回但没有历史曝光:未观察样本,不称为真实负反馈。
