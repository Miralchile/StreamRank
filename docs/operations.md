# 运行与故障处理手册

## 本地验收

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[ml,serving,streaming,dev]'
make verify
```

`make verify` 依次执行:Ruff lint/format check、完整单元与 API 测试、真实小规模日志冒烟
实验(`rank-smoke`)和 Python 静态编译。任一步失败都会返回非零退出码。

## 服务启动

内存状态(本地开发):

```bash
STREAMRANK_STATE_BACKEND=memory make serve-real
```

完整栈(Redis 状态 + Kafka 事件 + 独立 consumer):

```bash
docker compose up -d --build
open http://localhost:18000/
```

探针:

- `/health`:进程、deployment、状态后端和最近降级原因;
- `/ready`:catalog 为空时返回 503;
- `/metrics`:请求数、失败数、降级数、消费延迟、p50/p95/p99。

## 发布与回滚

一次发布必须绑定完整 manifest:模型、校准器、特征 schema、item index、重排策略和统一
`compatibility_key`。manifest 同时保存五个 descriptor 的 SHA-256;descriptor 还绑定
catalog 与模型 checkpoint 的摘要。任何组件缺失、摘要错误、版本或 key 不一致时拒绝激活。

标准发布流程:

1. `make rank-real` 产出新的胜出模型 artifact;
2. 需要调整重排权重时,修改 `configs/serving_policy.json`(显式声明,不伪装成调参结果);
3. `make build-deployment` 重建五个 descriptor 与 manifest,构建内含完整校验;
4. `make benchmark-real` 生成与新 deployment id 绑定的回归基准;
5. 启动/`ManifestStore.activate` 原子替换后检查 `/ready` 与核心推荐 case;
6. 失败时用上一份完整 manifest 整体回滚,不能单独回滚模型。

## 故障场景

| 场景 | 预期行为 | 观测指标 |
|---|---|---|
| 用户看完全部 catalog | 返回空列表,不偷偷推荐已看物品 | `degraded_total`, `no_unseen_candidates` |
| 重复事件 | Redis Lua 原子拒绝第二次状态更新 | sync `/events` 的 `deduplicated=true`;Kafka event id |
| Kafka 毒消息 | 先写入 `<topic>.dlq` 再提交源 offset | consumer error log、DLQ offset |
| Redis 启动失败 | Redis 模式启动失败,避免静默切换导致状态不一致 | 容器启动日志、readiness |
| manifest 不兼容 | 激活前抛错,旧 manifest 保持不变 | 发布日志 |
| 模型/索引不可用 | 使用预先验证的整套 fallback manifest | deployment id、降级计数 |
| 消费积压 | 在线特征保持旧版本并暴露 lag | `consumer_lag_ms` |

## 压测

```bash
make benchmark-real
```

这是单进程、真实 5k 队列 catalog 的本地回归基准,只验证明显性能退化。真实容量结论需要
使用真实特征后端、容器资源限制和独立压测机。

## 正式实验纪律

- 最终测试只运行一次;随机日志只做诊断;
- 报告必须同时保存 config、数据摘要、模型、选型历史和限制说明;
- 不把标准/随机日志混合后的指标包装成低偏差或无偏结果;
- 重排权重是显式产品参数,改动走 `serving_policy.json` 与 manifest 重建,不口头覆盖。
