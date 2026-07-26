# 验证记录

## 2026-07-26 单主线收敛后验证

本次重构删除旧五阶段控制面实验线(移入 `_to_delete/`),收敛为
"序列精排 → manifest绑定serving → 前端漏斗" 单主线,并做了如下验证
(云端环境无 torch/fastapi,相关测试按既有 skip 守卫跳过):

- `ruff check` 与 `ruff format --check`:通过;
- `python -m compileall src scripts tests`:通过;
- 无 torch/fastapi 依赖的单元测试(engine trace、serving build、manifest、
  scoring、metrics、audit、prepare、download、streaming):通过;
- `streamrank build-deployment`:用 `configs/serving_policy.json` 重建全部
  descriptor 与 manifest,`DeploymentBundle.load` 校验通过(含 5k catalog、
  artifact.json、deepfm.pt 的 SHA-256);
- 既有回归基准 `artifacts/benchmarks/kuairand-pure-sample.json` 的
  deployment_id 与重建后 manifest 一致。

**待本机复验**(需要完整 ml/serving 依赖):

```bash
pip install -e '.[ml,serving,streaming,dev]'
make verify          # lint + 全部测试(含API与torch用例) + rank-smoke
docker compose up -d --build   # 四服务集成与前端漏斗人工检查
```

## 2026-07-17 Docker 集成验收(历史记录,早于单主线收敛)

`docker compose up -d --build` 后四服务(api/redis/redpanda/feature-consumer)
持续运行,已验证:健康与就绪探针、带召回源的推荐返回、manifest descriptor
SHA-256 校验、`API → Redpanda → consumer → Redis Lua` 端到端幂等、毒消息进入
DLQ、`/metrics` 延迟与lag指标。当时 serving 绑定 demo 目录;该结论证明工程链路
可运行,不代表真实数据模型效果,也不代表线上 uplift。
