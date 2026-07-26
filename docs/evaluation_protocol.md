# 评测协议

本项目只有一条评测主线:`focused/` 序列精排实验。协议由
`configs/sequence_ranking_real.json` 固定,代码实现在 `focused/dataset.py` 与
`focused/runner.py`。

## 时间隔离

```text
Train        standard日志, event_time < 04-22        训练参数
Validation   standard日志, 04-22 ~ 04-30             早停与三模型选型
Final test   standard日志, 05-01 ~ 05-08             冻结后只评一次
Random diag  random日志,  event_time < 05-09         偏差诊断, 不参与任何选择
```

词表只用 standard 训练段构建;0 为 padding、1 为 OOV。验证集与测试集不参与词表、
归一化统计或早停以外的任何拟合。

## 请求语义

KuaiRand-Pure 的 request 字段不足以还原真实 slate,因此使用固定 1000ms 时间桶近似请求边界:
同桶内所有曝光先全部生成样本(共享桶前的历史与统计),再统一消费反馈更新用户历史、
用户/物品统计。这避免了同请求内的标签泄漏,但只是近似,不声称还原真实请求分组。

## 选型规则

- 早停与三模型对照均使用同一标量:验证集上 `is_click` 与 `long_view` 的用户 GAUC 均值;
- `is_like` 是辅助任务、`is_hate` 是稀疏风险诊断,均不参与早停或选型;
- 三组模型共享时间切分、词表、特征、任务权重与训练预算,MMoE 不因参数更多而豁免。

## 多任务指标

每个任务报告 ROC-AUC、PR-AUC、LogLoss、Brier、ECE、正样本数;主任务另报 GAUC:

```text
GAUC = Σ valid_user impressions(user) × AUC(user) / Σ valid_user impressions(user)
```

仅保留同时有正负样本的用户。稀有标签在部分切片上没有双类样本时,指标显式置为 null,
不允许 NaN 混入报告。

## 统计支撑

模型对照差距用两种互补方式量化(`scripts/compare_models.py`):用户级配对
bootstrap(重采样评测用户,B=2000,报告选型GAUC差距的95%百分位区间与
P(Δ≤0)),量化单次训练下评测队列的抽样不确定性;多seed重跑汇总
(`--seed/--output-dir` 重训全部模型)量化重训练方差与排名稳定性。二者都不度量
线上收益。

## 召回评测

召回阶段单独评测(`evaluation/recall.py`):检索器只用截止时间前的日志拟合;
每个用户的查询状态是其截止前轨迹,目标是评测窗口内首次长播且从未交互过的物品;
候选宇宙限定为截止前观测到的目录,排除已看物品,与serving引擎一致。报告
Recall@K、HitRate@K与目录覆盖率,按召回源与RRF融合分别汇总。该协议度量的是
日志候选池内的next-positive检索质量,未做曝光去偏。

## 随机日志

随机曝光切片只用于与标准日志对比预测与校准质量,是 logging-policy 偏差的诊断证据。
它不参与训练、早停或选型;其指标差距不能解释为任意新策略的无偏 OPE,也不等于线上收益。

## 冷启动与序列边界

Pure 的历史仅包含候选池内交互,不能称为完整用户兴趣序列;严格的长序列结论需要在
KuaiRand-1K 上复验。未见 user/item 通过 OOV embedding 处理,serving 侧同样如此。
