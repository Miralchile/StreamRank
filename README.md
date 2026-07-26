# StreamRank

一个**单主线、可复现、可展示**的短视频推荐系统项目:在真实快手 KuaiRand 日志上做
多目标序列精排实验,把离线胜出模型绑定进多阶段在线服务,并用前端完整展示
数据 → 训练 → 验证 → 在线推荐的全过程。

```text
KuaiRand日志
  → 时间正确的用户历史与动态特征
  → DeepFM式基线 / DIN / DIN+MMoE 同协议对照
  → 标准曝光测试 + 随机曝光偏差诊断
  → 胜出模型artifact(manifest绑定, SHA-256校验)
  → ItemCF/热门召回 → RRF融合 → DeepFM多目标精排 → 多样性重排
  → 前端漏斗展示 + 反馈事件经Kafka幂等更新Redis在线状态
```

整个仓库遵循最小化原则:只保留一个研究问题、一条训练主链路、一个评测协议和
一个可部署产物;每个在线组件(召回、融合、精排、校准、重排)都由同一个
deployment manifest 版本化绑定。

## 研究问题

> 在真实短视频多反馈日志上,候选相关序列建模和多任务专家结构,是否比非序列特征交叉
> 更有效,并能否在日志策略变化下保持稳定?

任务角色固定为:

- `is_click`(点击或有效播放)与 `long_view`:主任务,决定早停和模型选择;
- `is_like`:辅助任务;
- `is_hate`:稀疏风险诊断,不决定模型选择。

## 为什么使用 KuaiRand

KuaiRand来自快手真实推荐日志,包含标准曝光、随机干预曝光、时间戳和多种反馈信号。
本项目使用Pure完成多目标排序和偏差诊断;Pure仅保留候选池内视频,因此其历史不被称为
完整用户兴趣序列。严格长序列结论必须在KuaiRand-1K上补充验证。

主实验不使用跨月聚合的 `video_features_statistic.csv`,也不将随机曝光诊断解释为
任意新策略的无偏OPE。

## 一键运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[ml,serving,streaming,dev]'

# lint + 测试 + 真实小规模日志冒烟实验
make verify

# 下载官方数据 → 确定性5,000用户队列 → 正式训练与诊断
make rank-download
make rank-prepare
make rank-real
```

主要产物:

```text
artifacts/sequence-ranking-real/
├── deepfm.pt / din.pt / din_mmoe.pt
├── artifact.json       # 胜出模型、词表、归一化参数和checkpoint
└── report.json         # 全模型、全任务、分日志策略指标
```

启动完整可展示系统(FastAPI + Redis + Redpanda + 独立特征consumer):

```bash
docker compose up -d --build
open http://localhost:18000/
```

前端按 HR/面试官可读的方式展示三段证据:真实训练数据与时间切分、三模型训练对照与
选型、离线测试与分布变化诊断;最后是在线推荐演示——选择真实用户调用 `/recommend`,
页面用**多阶段漏斗**展示每一层候选数(ItemCF/热门召回 → RRF融合 → DeepFM精排 →
多样性重排 Top-K),并可模拟一次长播反馈:事件发布到 Kafka/Redpanda,以同一 event id
幂等写入在线状态,再刷新推荐观察变化。

## 模型对照

| 实验 | 序列注意力 | 多任务结构 | 回答的问题 |
|---|---:|---|---|
| DeepFM式基线 | 否 | Shared-bottom | 普通特征交叉能达到什么水平? |
| DIN | 是 | Shared-bottom | 候选相关兴趣是否有效? |
| DIN+MMoE | 是 | 4 Experts/Gates | 任务共享与分离是否继续提升? |

三组模型使用相同时间切分、词表、特征、任务权重和早停指标。MMoE不会仅凭参数更多
被宣布有效;必须同时报告参数量、验证集选择结果和最终测试结果。

## 当前真实实验结果

固定seed为2026的5,000用户队列包含:

- 482,472条真实交互;
- 263,206条标准曝光,219,266条随机曝光;
- 7,581个物品;
- 正式训练208,943行、验证29,522行、测试24,741行。

当前最终测试结果:

| 模型 | Click AUC / GAUC | Long-view AUC / GAUC | 验证主任务GAUC |
|---|---|---|---:|
| DeepFM式基线 | 0.7266 / 0.6171 | 0.7216 / 0.6241 | **0.6395** |
| DIN | 0.7211 / 0.6101 | 0.7188 / 0.6197 | 0.6342 |
| DIN+MMoE | 0.7224 / 0.6129 | 0.7177 / 0.6178 | 0.6322 |

因此当前胜出模型是DeepFM式基线,也是在线服务绑定的精排模型。项目**没有**声称DIN/MMoE
在Pure上获得提升。该结果支持"模型复杂度必须匹配数据可观测性"的判断,但尚不能证明DIN
本身无效;下一步需在具有严格序列日志的KuaiRand-1K上复验。

随机曝光上主任务指标明显下降,这说明标准日志效果包含logging-policy分布信息。它是偏差
诊断证据,不是新策略在线收益。

完整数字见
[`artifacts/sequence-ranking-real/report.json`](artifacts/sequence-ranking-real/report.json)。

## 代码结构

```text
src/streamrank/
├── focused/            # 唯一实验主线:时间切分、序列样本、三模型训练与选型导出
├── engine.py           # 在线多阶段引擎:召回→融合→精排→重排,带per-stage trace
├── retrieval/          # ItemCF、热门/类目召回与RRF融合
├── ranking/            # PolicyScorer、校准接口、FocusedServingRanker(加载胜出checkpoint)
├── rerank/             # 多样性约束重排
├── serving/            # FastAPI应用、deployment manifest校验、构建工具
├── streaming/          # Kafka生产者与幂等特征consumer(DLQ)
├── online/             # 内存/Redis在线状态
└── data/               # KuaiRand下载、确定性队列准备、标签审计、CSV加载

configs/
├── sequence_ranking_smoke.json   # 冒烟实验
├── sequence_ranking_real.json    # 正式实验
└── serving_policy.json           # 显式声明的重排权重与多样性约束

deployments/            # manifest + 5组件descriptor(含SHA-256与catalog摘要)
frontend/               # 三段式展示页 + 在线推荐漏斗
scripts/                # run_sequence_ranking.py, benchmark_engine.py
```

## 能与不能声称的结论

可以声称:

- 在真实快手交互日志上构建时间正确的多目标排序实验;
- 完成DeepFM式、DIN和DIN+MMoE的同协议对照;
- 将标准曝光最终测试与随机曝光偏差诊断分开;
- 发现复杂序列模型在Pure的不完整历史上没有超过强非序列基线;
- 将离线胜出的DeepFM checkpoint接入在线推荐服务,并通过前端漏斗展示
  召回/融合/精排/重排与反馈更新链路;
- 召回→精排→重排→在线反馈的工程与评测方法可平移到搜索、广告排序场景。

不能声称:

- DIN+MMoE带来线上CTR提升;
- 随机曝光指标等价于任意策略的无偏OPE;
- 重排权重经过了日志数据上的策略搜索(它们是显式pre-registered产品参数);
- 当前公共数据实验等价于生产A/B测试;
- 反馈演示等价于真实用户对新策略的行为响应。

更详细的语义见 [`docs/data_contract.md`](docs/data_contract.md)、
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)、
[`docs/architecture.md`](docs/architecture.md) 和 [`docs/operations.md`](docs/operations.md)。
