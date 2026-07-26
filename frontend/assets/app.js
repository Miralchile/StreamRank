const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const MODEL_LABELS = {
  deepfm: "特征交互排序（DeepFM）",
  din: "兴趣序列排序（DIN）",
  din_mmoe: "多任务专家排序（DIN + MMoE）",
};

const MODEL_DESCRIPTIONS = {
  deepfm: "组合用户、视频、场景与历史统计，预测用户反馈。该版本在当前验证集上表现最好。",
  din: "结合近期观看序列，判断候选视频是否符合用户当前兴趣。",
  din_mmoe: "联合学习有效播放、长播、点赞和负反馈四个目标。",
};

const state = {
  focused: null,
  health: null,
  profiles: null,
  selectedModel: null,
  lastRecommendation: null,
};

const formatNumber = (value, digits = 0) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toLocaleString("zh-CN", { maximumFractionDigits: digits });
};

const formatPercent = (value, digits = 1) => {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
};

const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[char]);

async function request(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12_000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const type = response.headers.get("content-type") || "";
    const payload = type.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(payload?.detail || `请求失败（${response.status}）`);
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求超时，请检查服务状态");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = "toast"; }, 3200);
}

function renderDatasetView() {
  const { dataset, experiment } = state.focused;
  const form = new FormData($("#datasetControls"));
  const cohort = Math.max(500, Math.min(5000, Number(form.get("cohort_size")) || 5000));
  const ratio = cohort / 5000;
  const policy = String(form.get("logging_policy"));
  const target = String(form.get("target"));
  const historyLength = Math.max(1, Math.min(100, Number(form.get("history_length")) || 50));
  const policyRows = dataset.rows_by_logging_policy || {};
  const sourceRows = policy === "all"
    ? Number(dataset.interactions)
    : Number(policyRows[policy] || 0);
  const policyLabels = { standard: "标准推荐日志", random: "随机曝光日志", all: "全部日志" };
  const targetLabels = { is_click: "有效播放 / 点击", long_view: "长播", is_like: "点赞", is_hate: "负反馈" };
  const rates = dataset.label_positive_rates || {};
  $("#viewRows").textContent = formatNumber(Math.round(sourceRows * ratio));
  $("#viewPolicy").textContent = `${formatNumber(cohort)} 用户 · ${policyLabels[policy]}`;
  $("#viewPositiveRate").textContent = formatPercent(rates[target], 2);
  $("#viewTarget").textContent = targetLabels[target];
  $("#viewItems").textContent = formatNumber(dataset.items);
  $("#viewHistory").textContent = `${historyLength} 条`;

  const rows = experiment.dataset.rows;
  $("#trainRows").textContent = `${formatNumber(Math.round(rows.train * ratio))} 行`;
  $("#validationRows").textContent = `${formatNumber(Math.round(rows.validation * ratio))} 行`;
  $("#testRows").textContent = `${formatNumber(Math.round(rows.test * ratio))} 行`;
}

function modelFlow(name) {
  const sequence = name !== "deepfm";
  const mmoe = name === "din_mmoe";
  const nodes = [
    ["01 / 输入", "用户与视频信息", "用户、视频、场景与历史统计"],
    ["02 / 匹配", sequence ? "判断近期兴趣" : "学习偏好关系", sequence ? "当前视频是否符合最近兴趣" : "什么用户喜欢什么视频"],
    ["03 / 学习", mmoe ? "多目标分别学习" : "共享用户偏好", mmoe ? "不同反馈使用不同专家" : "多个反馈共同训练"],
    ["04 / 输出", "预测用户反馈", "有效播放 · 长播 · 点赞 · 负反馈"],
  ];
  return nodes.map(([step, title, detail]) => `<article class="flow-node"><small>${step}</small><strong>${title}</strong><span>${detail}</span></article>`).join("");
}

function selectModel(name) {
  const report = state.focused.experiment;
  const model = report.models[name];
  state.selectedModel = name;
  $$(".model-tab").forEach((tab) => {
    const active = tab.dataset.model === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $("#selectedModelName").textContent = MODEL_LABELS[name] || name;
  $("#selectedModelParams").textContent = `${formatNumber(model.parameters)} 参数`;
  $("#algorithmFlow").innerHTML = modelFlow(name);
  $("#modelExplanation").textContent = MODEL_DESCRIPTIONS[name] || "";
  $("#bestEpoch").textContent = `BEST EPOCH ${model.best_epoch}`;
  $("#selectionScore").textContent = Number(model.validation_selection_gauc).toFixed(4);
  const history = model.history || [];
  const values = history.map((item) => Number(item.selection_gauc)).filter(Number.isFinite);
  const min = Math.min(...values, 0.5);
  const max = Math.max(...values, 0.7);
  $("#epochChart").innerHTML = history.map((item) => {
    const score = Number(item.selection_gauc);
    const height = 18 + ((score - min) / Math.max(.001, max - min)) * 120;
    return `<div class="epoch${item.epoch === model.best_epoch ? " best" : ""}" title="Epoch ${item.epoch}: ${score.toFixed(4)}"><i style="height:${height}px"></i><span>${item.epoch}</span></div>`;
  }).join("");
}

function comparisonRow(label, task, standard, random) {
  const std = Number(standard?.[task]?.roc_auc);
  const rnd = Number(random?.[task]?.roc_auc);
  return `
    <div class="policy-group">
      <div class="policy-label"><span>${label} · 标准推荐</span><b>${std.toFixed(4)}</b></div>
      <div class="policy-bar"><i style="width:${std * 100}%"></i></div>
    </div>
    <div class="policy-group random">
      <div class="policy-label"><span>${label} · 随机曝光</span><b>${rnd.toFixed(4)}</b></div>
      <div class="policy-bar"><i style="width:${rnd * 100}%"></i></div>
      <div class="policy-task">分布变化后下降 ${(std - rnd).toFixed(4)}</div>
    </div>`;
}

function renderFocused() {
  const { dataset, experiment, serving } = state.focused;
  const winner = experiment.winner;
  const winnerModel = experiment.models[winner];
  const test = winnerModel.metrics.test;
  const random = winnerModel.metrics.random_diagnostic;
  $("#datasetProvenance").innerHTML = `REAL DATASET <b>${escapeHtml(dataset.dataset)}</b> · ${formatNumber(dataset.interactions)} 交互 · ${formatNumber(dataset.selected_users)} / ${formatNumber(dataset.source_users)} 用户 · ${formatNumber(dataset.items)} 视频 · ${escapeHtml(dataset.license)} · MD5 ${escapeHtml(dataset.archive_md5)}`;
  $("#heroRows").textContent = formatNumber(dataset.interactions);
  $("#heroWinner").textContent = MODEL_LABELS[winner] || winner;
  $("#heroAuc").textContent = Number(test.long_view.roc_auc).toFixed(3);

  $("#modelTabs").innerHTML = Object.entries(experiment.models).map(([name, model]) => `
    <button class="model-tab" type="button" role="tab" aria-selected="false" data-model="${escapeHtml(name)}">
      <b>${escapeHtml(MODEL_LABELS[name] || name)}</b>
      <small>验证 GAUC ${Number(model.validation_selection_gauc).toFixed(4)}</small>
    </button>`).join("");
  $$(".model-tab").forEach((tab) => tab.addEventListener("click", () => selectModel(tab.dataset.model)));
  selectModel(winner);

  $("#winnerName").textContent = MODEL_LABELS[winner] || winner;
  $("#winnerReason").textContent = `它在独立验证集上的点击与长播用户 GAUC 均值最高（${Number(winnerModel.validation_selection_gauc).toFixed(4)}），因此进入最终测试；测试集没有参与版本选择。`;
  const alignment = $("#servingAlignment");
  alignment.textContent = serving.offline_winner_bound ? "已接入在线服务" : "在线接入待完成";
  alignment.classList.toggle("ok", serving.offline_winner_bound);

  const clickAuc = Number(test.is_click.roc_auc);
  const longAuc = Number(test.long_view.roc_auc);
  $("#clickAuc").textContent = clickAuc.toFixed(4);
  $("#longAuc").textContent = longAuc.toFixed(4);
  $("#positiveRows").textContent = formatNumber(test.long_view.positive_rows);
  $("#clickMeaning").textContent = `随机抽取一条正样本和一条负样本，约有 ${(clickAuc * 100).toFixed(1)}% 的概率把正样本排得更高。`;
  $("#longMeaning").textContent = `模型对长播与未长播样本具有 ${(longAuc * 100).toFixed(1)}% 的成对排序能力。`;
  $("#policyCompare").innerHTML = comparisonRow("有效播放", "is_click", test, random) + comparisonRow("长播", "long_view", test, random);

  const steps = [
    ["01", "真实数据训练", true],
    ["02", "独立测试验证", true],
    ["03", "胜出模型在线绑定", serving.offline_winner_bound],
    ["04", "反馈更新在线特征", true],
  ];
  $("#loopStatus").innerHTML = steps.map(([number, label, done]) => `<article class="loop-step ${done ? "done" : "pending"}"><small>${number} / ${done ? "READY" : "PENDING"}</small><strong>${label}</strong></article>`).join("");
  renderDatasetView();
}

async function loadHealth() {
  const health = await request("/health");
  state.health = health;
  $("#deploymentId").textContent = health.deployment_id;
  const query = $("input[name=query_time_ms]");
  query.value = String(Number(health.fit_cutoff_ms) + 1);
  $("#recommendForm button[type=submit]").disabled = false;
  $("#recommendState").textContent = "服务就绪；结果来自当前在线deployment";
}

const RANKER_STAGE_NAMES = {
  deepfm: "DeepFM 多目标打分",
  din: "DIN 多目标打分",
  din_mmoe: "DIN+MMoE 多目标打分",
  heuristic: "启发式多目标打分",
};

function renderPipeline(data) {
  const funnel = $("#pipelineFunnel");
  const pipeline = data.pipeline;
  if (!pipeline || !Array.isArray(pipeline.stages)) {
    funnel.hidden = true;
    return;
  }
  const byStage = Object.fromEntries(pipeline.stages.map((item) => [item.stage, Number(item.count) || 0]));
  const rankerName = RANKER_STAGE_NAMES[data.ranker?.architecture]
    || RANKER_STAGE_NAMES[data.ranker?.kind]
    || "多目标打分";
  const columns = [
    { kind: "recall", cells: [
      { eyebrow: "召回 A", name: "ItemCF 协同过滤", count: byStage.recall_itemcf },
      { eyebrow: "召回 B", name: "热门 / 类目", count: byStage.recall_popularity },
    ] },
    { kind: "stage", cells: [{ eyebrow: "融合", name: "RRF 去重合并", count: byStage.fusion }] },
    { kind: "stage", cells: [{ eyebrow: "精排", name: rankerName, count: byStage.ranking }] },
    { kind: "stage", cells: [{ eyebrow: "重排", name: "多样性约束 Top-K", count: byStage.rerank }] },
  ];
  const maxCount = Math.max(1, ...pipeline.stages.map((item) => Number(item.count) || 0));
  funnel.hidden = false;
  $("#funnelMeta").textContent =
    `可推荐候选池 ${formatNumber(pipeline.catalog_size)} · 已看历史过滤 ${formatNumber(pipeline.excluded_seen)} · 每阶段候选数`;
  $("#funnelStages").innerHTML = columns.map((column) => `
    <div class="funnel-col${column.kind === "recall" ? " recall" : ""}">
      ${column.cells.map((cell) => `
        <article class="funnel-stage">
          <small>${escapeHtml(cell.eyebrow)}</small>
          <strong>${formatNumber(cell.count)}</strong>
          <span>${escapeHtml(cell.name)}</span>
          <i style="width:${Math.max(8, Math.round((Number(cell.count) || 0) / maxCount * 100))}%"></i>
        </article>`).join("")}
    </div>`).join("");
}

function renderRecommendation(data) {
  state.lastRecommendation = data;
  $("#feedbackButton").disabled = !data.items.length;
  renderPipeline(data);
  $("#recommendResults").innerHTML = data.items.length ? data.items.map((item, index) => {
    const features = item.features || {};
    const source = escapeHtml((item.sources || []).join(" + ") || "fallback");
    const longView = Number(features.long_view);
    const category = features.category ? `类目 ${escapeHtml(String(features.category).replace(/^tag:/, ""))}` : "类目未知";
    return `
      <article class="result-item">
        <span>#${String(index + 1).padStart(2, "0")}</span>
        <div><strong>视频 ${escapeHtml(item.item_id)}</strong><small>${source} · ${category} · 长播概率 ${formatPercent(longView, 1)}</small></div>
        <code>${formatNumber(item.score, 5)}</code>
      </article>`;
  }).join("") : '<p class="empty">当前用户没有可返回的未看候选。</p>';
  const ranker = data.ranker?.architecture
    ? `${data.ranker.architecture} 在线排序`
    : `${data.ranker?.kind || "unknown"} 在线排序`;
  $("#recommendState").textContent = data.personalization_mode === "history-aware"
    ? `已使用 ${data.history_size} 条历史行为 · ${ranker} · ${data.deployment_id}`
    : `无历史用户，使用热门回退 · ${ranker} · ${data.deployment_id}`;
}

async function submitRecommendation(form) {
  const params = new URLSearchParams({
    top_k: form.get("top_k"),
    query_time_ms: form.get("query_time_ms"),
    tab: form.get("tab") || "0",
  });
  const data = await request(`/recommend/${encodeURIComponent(form.get("user_id"))}?${params}`);
  renderRecommendation(data);
  return data;
}

function renderProfiles(payload) {
  state.profiles = payload;
  const select = $("#servingUserSelect");
  const profiles = payload.profiles || [];
  select.innerHTML = profiles.length
    ? profiles.map((profile) => `<option value="${profile.user_id}">用户 ${profile.user_id}｜历史 ${profile.history_size} 条</option>`).join("")
    : '<option value="">没有可用用户画像</option>';
  if (payload.default_user_id != null) select.value = String(payload.default_user_id);
  renderSelectedProfile();
}

function renderSelectedProfile() {
  const select = $("#servingUserSelect");
  const profile = state.profiles?.profiles?.find((item) => String(item.user_id) === select.value);
  const summary = $("#selectedProfileSummary");
  if (!profile) {
    summary.textContent = "当前没有可用用户历史。";
    return;
  }
  const category = profile.preferred_category
    ? `主要偏好：类目 ${String(profile.preferred_category).replace(/^tag:/, "")}`
    : "主要偏好：暂无明确类目";
  summary.textContent = `${profile.history_size} 条可用历史 · ${category}`;
}

async function loadAll(showToast = false) {
  try {
    const [focused, profiles] = await Promise.all([
      request("/api/focused"),
      request("/api/serving-users"),
      loadHealth(),
    ]);
    state.focused = focused;
    renderProfiles(profiles);
    renderFocused();
    $("#sidePulse").classList.add("ok");
    $("#sideStatus").textContent = "实验数据已核验";
    $("#sideBackend").textContent = "真实数据 · 正式报告";
    $("#topStatus").textContent = "真实实验已加载";
    if (showToast) toast("实验数据已刷新");
  } catch (error) {
    $("#sideStatus").textContent = "加载失败";
    $("#topStatus").textContent = "实验产物不可用";
    toast(error.message, true);
  }
}

$("#datasetControls").addEventListener("submit", (event) => {
  event.preventDefault();
  renderDatasetView();
  toast("训练数据视图已更新；正式实验结果未被改写");
});

$("#servingUserSelect").addEventListener("change", renderSelectedProfile);

$("#recommendForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  $("#feedbackButton").disabled = true;
  $("#recommendState").textContent = "正在调用当前在线推荐服务…";
  try {
    await submitRecommendation(form);
    $("#feedbackState").textContent = "可对首个推荐结果模拟长播反馈。";
  } catch (error) {
    $("#recommendState").textContent = error.message;
    toast(error.message, true);
  } finally {
    button.disabled = false;
    $("#feedbackButton").disabled = !(state.lastRecommendation?.items || []).length;
  }
});

$("#feedbackButton").addEventListener("click", async () => {
  const formElement = $("#recommendForm");
  const form = new FormData(formElement);
  const item = state.lastRecommendation?.items?.[0];
  if (!item) return;
  const button = $("#feedbackButton");
  button.disabled = true;
  $("#feedbackState").textContent = "正在写入长播反馈…";
  try {
    const eventTime = Number(form.get("query_time_ms")) + 1;
    await request("/events", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Event-ID": `frontend-loop-${form.get("user_id")}-${item.item_id}-${eventTime}`,
      },
      body: JSON.stringify({
        user_id: Number(form.get("user_id")),
        item_id: Number(item.item_id),
        event_time_ms: eventTime,
        is_click: 1,
        long_view: 1,
        is_like: 0,
        is_hate: 0,
        tab: Number(form.get("tab") || 0),
        category: String(item.features?.category || "UNKNOWN"),
        logging_policy: "frontend_feedback_demo",
        request_id: `frontend-${Date.now()}`,
      }),
    });
    formElement.querySelector("input[name=query_time_ms]").value = String(eventTime + 1);
    await submitRecommendation(new FormData(formElement));
    $("#feedbackState").textContent = `已消费视频 ${item.item_id} 的长播反馈，并刷新推荐。`;
  } catch (error) {
    $("#feedbackState").textContent = error.message;
    toast(error.message, true);
  } finally {
    button.disabled = !(state.lastRecommendation?.items || []).length;
  }
});

$("#refreshAll").addEventListener("click", () => loadAll(true));
$("#menuButton").addEventListener("click", () => {
  const sidebar = $(".sidebar");
  const open = !sidebar.classList.contains("open");
  sidebar.classList.toggle("open", open);
  $("#menuButton").setAttribute("aria-expanded", String(open));
});

const sections = ["data", "training", "evidence"].map((id) => document.getElementById(id));
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    $$(".nav-link").forEach((link) => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
  });
}, { rootMargin: "-25% 0px -65% 0px" });
sections.forEach((section) => observer.observe(section));

loadAll();
