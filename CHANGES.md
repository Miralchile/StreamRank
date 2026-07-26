# 2026-07-26 单主线收敛重构

目标:符合搜广推岗位展示需要、可完整前后端演示,并遵循最小化原则——
一个研究问题、一条训练主链路、一个评测协议、一个可部署产物。

## 删(全部移入 `_to_delete/`,未销毁,确认后可整目录删除)

旧"五阶段控制面"实验线与序列精排主线重复,且其唯一对 serving 的贡献
(重排权重)实际是 pre-registered fallback 值,故整线退役:

- `src/streamrank/experiment.py`、`training/`、`replay/`、`config.py`
- `src/streamrank/data/`:`pit.py`、`cold_start.py`、`evaluation_queries.py`、`splits.py`
- `src/streamrank/evaluation/`:`bootstrap.py`、`ranking.py`、`retrieval.py`
- `src/streamrank/retrieval/two_tower.py`(未接入 serving 的离线实验模块)
- `scripts/train_demo_models.py`、`artifacts/demo/`、`artifacts/experiments/`
- `configs/`:`default.json`、`demo_experiment.json`、两个 kuairand 五阶段实验配置
- `tests/`:`test_experiment`、`test_pit_features`、`test_protocol_boundaries`、`test_retrieval`
- 缓存噪声:`focused/__pycache__/`、`streamrank.egg-info/`

## 改

- **重排权重显式化**:新增 `configs/serving_policy.json`;`build-deployment` 从它读取
  score 权重与多样性约束,不再依赖五阶段报告(`serving/build.py`、`cli.py`)。
  rerank descriptor 的 `weight_source` 随之指向 policy 配置,manifest 已重建并通过校验;
  deployment_id 不变,既有压测报告仍然匹配。
- **API 精简**:移除前端未使用的 `/api/project`、`/api/dataset`、`/api/experiment`、
  `/api/benchmark`;保留 `/api/focused`、`/api/serving-users`、`/health`、`/ready`、
  `/manifest`、`/recommend`、`/events`、`/metrics`。
- **漏斗数据**:`engine.recommend(..., trace=)` 记录各阶段候选数,
  `/recommend` 返回 `pipeline` 字段(catalog_size / excluded_seen / 五个阶段计数)。
- **一致性修复**:Makefile 部署与压测目标由 sample catalog 对齐到实际绑定的 5k catalog;
  Dockerfile / docker-compose / .env.example 移除五阶段引用;`pyproject` 移除未用的
  faiss 依赖;修复 `engine.py`/`serving_ranker.py` 存量 lint 违规;`.gitignore` 增加
  `_to_delete/`。

## 增

- 前端"在线推荐"新增**多阶段漏斗**:ItemCF/热门召回 → RRF 融合 → DeepFM 精排 →
  多样性重排,展示每层候选数与收敛比例(`index.html`/`app.js`/`styles.css`)。
- 新测试:engine trace 断言、policy 构建(含权重符号校验)、API pipeline 字段与
  退役端点 404 断言。

## 查(验证)

- ruff check / format、compileall:通过;无 torch/fastapi 依赖的 22 项测试:通过
  (11 项按既有守卫跳过,待本机 `make verify` 复验);
- manifest 重建后 `DeploymentBundle.load` 全量校验通过(catalog/artifact/checkpoint SHA-256);
- 真实前端在带打桩后端的浏览器中渲染验证:漏斗、推荐列表、三段证据页均正常,无 JS 报错。

## 建议(未代做)

- 目录尚无 git:建议 `git init` 后首次提交,当前 `.gitignore` 会排除大数据与 artifacts;
- 本机执行 `make verify` 与 `docker compose up -d --build` 做最终集成复验;
- 确认无误后删除 `_to_delete/`。

# 2026-07-26 完善轮:测试闭环、CI、统计支撑、召回评测

按"完整实验+部署"标准补齐四项:

1. **测试闭环**:33/33 首次在完整依赖环境执行并通过;修复 API 测试守卫
   (RuntimeError)与 httpx→httpx2 依赖迁移。
2. **CI**:`.github/workflows/ci.yml` 跑 lint + 全部测试 + rank-smoke +
   compileall(CPU torch),README 加徽章。
3. **统计支撑**:`scripts/compare_models.py` 输出选型 GAUC 差距的用户级配对
   bootstrap 置信区间与多 seed 重跑汇总。结果实质改写了项目结论:测试集两两
   差距的 CI 全部跨零,且 3 个独立种子的胜者翻转(DeepFM 1 次、DIN 2 次),
   README 由"DeepFM 胜出"改为"三模型统计不可分、单次胜出由种子决定,部署
   沿用预注册协议在 seed=2026 选出的 checkpoint"。
4. **召回/重排评测**:`evaluation/recall.py` + `make recall-eval`,时间正确的
   next-positive 协议;量化三路召回与融合的 Recall@K/HitRate/覆盖率、重排的
   多样性-分数权衡;如实报告"朴素 RRF 低于单路 ItemCF"的负结果。


# 2026-07-26 注意力谱系扩展(LLM相关骨架的协议内检验)

- 新增三个小规模 Transformer 对照:SASRec式(因果自注意力)、BST式(候选参与
  序列注意力)、AutoInt式(特征域注意力,非序列),全部复用既有协议、参数量级
  与统计纪律;smoke/CI 同步覆盖六模型。
- 结果:序列自注意力无优势(SASRec 显著差于 DeepFM),非序列的特征域注意力
  3/3 seed 一致胜出(0.6431±0.0031)——"注意力放在特征交互而非序列上"是该
  数据上唯一有统计支撑的结构增益。
- 原生 macOS venv 建立,37 项测试原生全绿;修复 macOS /tmp 符号链接路径缺陷。
- 部署仍为 DeepFM checkpoint;切换到 AutoInt 需声明新选型协议后另行发布。


# 2026-07-26 发布:在线精排切换至 AutoInt

- 声明新选型协议(六模型注意力谱系、同验证集规则、seed=2026),胜出的 AutoInt
  checkpoint 提升为受版本管理的 serving 工件并重建 manifest;
- 原生冒烟 + 37 项测试 + 发布压测(QPS 93.7 / p95 121ms,较 DeepFM 的推理成本
  上升如实记录)+ docker 四服务重建;
- 前端与全部文档同步至六模型叙事;回滚路径为上一份完整 manifest。
