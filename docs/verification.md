# 验证记录

## 2026-07-26(第四轮)AutoInt 部署切换

- 按新声明协议(六模型、同验证集选型规则、seed=2026)将在线精排从 DeepFM 切换
  为 AutoInt:胜出 checkpoint 提升至 `artifacts/serving/autoint-v1/`(受版本管理),
  `build-deployment --artifact` 重建 manifest,model descriptor 为
  `focused-autoint-kuairand-pure-v1`,全部 SHA-256 校验通过。
- 原生冒烟:bundle 加载 + 48.2 万事件拟合 8.1s + AutoInt 端到端推荐,输出有限、
  trace 完整;37 项测试在新默认配置(六模型报告 + AutoInt manifest)下全绿。
- 发布压测重跑:QPS 93.7 / p50 82ms / p95 121ms / 0 错误——较 DeepFM
  (216 / 105ms)的推理成本上升如实记录,是结构收益与延迟的显式权衡。
- 前端"训练与选型"区切换为六模型对照报告,新增 SASRec/BST/AutoInt 的标签、
  说明与流程图;旧 DeepFM descriptor 保留作为回滚组件。

## 2026-07-26(第三轮)注意力谱系对照

- 新增 SASRec式/BST式/AutoInt式三个协议内对照模型;`FocusedRanker` 扩展为五种
  编码器路由,serving 侧同步支持部署任意架构。
- 本机建立原生 venv(python3.13 + macOS arm64 torch):**37 项测试首次在原生
  macOS 全部执行并通过**,并捕获一个仅在 macOS 暴露的真实缺陷(`/tmp` 为
  `/private/tmp` 符号链接导致 build.py 的 `relative_to` 失败,已修复)。
- 3 seed × 6 模型全部原生重训(同一环境保证可比):AutoInt式 3/3 seed 胜出,
  验证选型GAUC均值 0.6431±0.0031;SASRec式显著差于 DeepFM(bootstrap CI
  [+0.0028, +0.0152]);BST式与 DeepFM/DIN 不可分。逐seed报告与bootstrap汇总
  见 artifacts/transformer-comparison/comparison.json。
- 在线部署未变更(仍为预注册协议下的 seed-2026 DeepFM checkpoint);切换部署
  属于独立发布决策。

## 2026-07-26(第二轮)完善验证

- **全部 33 项测试首次在完整依赖环境实际执行并通过**(本机 docker 镜像,
  含 torch 与 fastapi;此前 API 与 torch 用例仅有 skip 守卫)。过程中修复了
  两个真实问题:starlette 新版 TestClient 缺依赖时抛 RuntimeError 而测试守卫
  只捕获 ImportError;dev extra 的 httpx 已被 starlette 的 httpx2 要求取代。
- **GitHub Actions CI 上线并通过**:CPU torch + lint + 33 项测试 + 真实日志
  冒烟实验 + compileall,首轮 2m08s 全绿(run 30196709351)。
- **模型对照补统计支撑**(`scripts/compare_models.py`,B=2000 用户级配对
  bootstrap):验证集 DeepFM−DIN+MMoE 差距 +0.0073,95% CI [+0.0015, +0.0135];
  DeepFM−DIN +0.0053,CI [−0.0010, +0.0115] 跨零;最终测试集三组两两差距 CI
  全部跨零。独立种子重训:2026 胜者 DeepFM(0.6395),2027 胜者 DIN(0.6340),
  2028 胜者 DIN(0.6397),排名翻转直接观测到;均值 0.6355/0.6360/0.6328,
  σ≈0.003–0.004。原计划 4 个额外种子,因 docker VM 中序列模型训练较原生慢约
  7 倍,在 2027/2028 完成后截断(2029/2030 未跑);3 个独立种子已足以支撑
  "统计不可分、胜出随种子翻转"的结论,README 已按此改写。
- **召回/重排量化评测**(`make recall-eval`,3,845 用户):ItemCF
  Recall@100=0.158 / HitRate@200=0.548 / 覆盖 81.4%;类目热门 0.076/0.335/71.1%;
  朴素 RRF 融合 0.139/0.524/81.5%,在所有 K 低于单路 ItemCF(弱源稀释,
  如实报告)。重排(200 用户,Top-20):类目数 7.1→17.8,最大类目占比
  54%→9.2%,保留 87.3% 分数、与纯分数序重合 44%。

## 2026-07-26(第一轮)单主线收敛验证

- ruff check / format、compileall 通过;无 torch/fastapi 依赖的 22 项测试通过;
- `streamrank build-deployment` 用显式 policy 配置重建全部 descriptor 与
  manifest,`DeploymentBundle.load` 校验通过(5k catalog、artifact、checkpoint
  的 SHA-256);
- 本机 `docker compose up -d --build` 四服务重建,`/health`、`/ready`、
  `/recommend`(含 pipeline 字段、DeepFM 绑定)线上验证通过,前端漏斗渲染无
  JS 报错。

## 2026-07-17 Docker 集成验收(历史记录)

四服务持续运行;健康/就绪探针、带召回源的推荐返回、manifest descriptor
SHA-256 校验、`API → Redpanda → consumer → Redis Lua` 端到端幂等、毒消息进入
DLQ、`/metrics` 指标均验证。当时 serving 绑定 demo 目录;结论证明工程链路可
运行,不代表真实数据模型效果,也不代表线上 uplift。
