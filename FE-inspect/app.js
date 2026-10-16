(async () => {
  "use strict";
  // A cached pre-localization HTML page may still load this script without the new helpers.
  if (!window.CoreInspectI18n) await import("./i18n.js?v=20261014-ux3");
  if (!window.CoreInspectViewers) await import("./viewers.js?v=20261014-ux3");
  const i18n = window.CoreInspectI18n, t = i18n.t, viewers = window.CoreInspectViewers;
  const $ = (selector) => document.querySelector(selector);
  const state = {
    base: localStorage.getItem("core-studio-base-url") || location.origin,
    token: sessionStorage.getItem("core-studio-token") || "",
    config: null, runs: [], selected: null, detail: null, spans: [], coverage: "not_recorded",
    storage: null, revision: null, tab: "flow", selection: null, collapsed: new Set(),
    zoom: 1, timelineZoom: 1, dataModes: new Map(), jsonOpen: new Map(), eventSearch: "", generation: 0, busy: false, traceBusy: false, runsBusy: false,
    runLimit: 200, runTotal: 0, runQuery: "", runQueryRevision: 0, instances: [], eventsMore: false, eventsBusy: false,
    lastDetailFetch: 0, connected: false, artifactRequest: 0, artifactContent: null, connectTimer: null, connectAttempts: 0, eventIndex: [],
  };
  const colors = { agent: "#7950d1", llm: "#147da2", tool: "#bd7b16", retrieval: "#169478", task: "#657b9e", operation: "#657b9e", run: "#4163e8" };
  const labels = { INITIALIZED: "Khởi tạo", PROFILING: "Chuẩn bị", READY: "Sẵn sàng", STAGE_ACTIVE: "Đang chạy", STAGE_TRANSITIONING: "Chuyển giai đoạn", DIALOGUE_COMPLETED: "Kết thúc hội thoại", ASSESSING: "Đánh giá", ENDING: "Đang kết thúc", REPORT_READY: "Hoàn tất", FAILED: "Lỗi", ASSESSMENT_FAILED: "Lỗi đánh giá", CANCELLED: "Đã hủy", EXPIRED: "Hết hạn", ARCHIVED: "Đã lưu", running: "Đang chạy", completed: "Hoàn tất", failed: "Lỗi", cancelled: "Đã hủy", abandoned: "Gián đoạn", timeout: "Timeout", queued: "Chờ chạy", degraded: "Suy giảm" };
  const complete = new Set(["REPORT_READY", "ARCHIVED", "completed"]);
  const failed = new Set(["FAILED", "ASSESSMENT_FAILED", "failed", "timeout", "abandoned", "degraded"]);
  const inactive = new Set(["CANCELLED", "EXPIRED", "cancelled"]);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const stringify = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const statusClass = (value) => complete.has(value) ? "completed" : failed.has(value) ? "failed" : inactive.has(value) || !value ? "" : "running";
  const statusLabel = (value) => t(labels[value] || value || "Chưa ghi trạng thái");
  function ms(value) {
    if (value === undefined || value === null || value === "") return null;
    if (typeof value === "number" || /^\d+(\.\d+)?$/.test(String(value))) {
      const number = Number(value); return Number.isFinite(number) ? (number < 1e11 ? number * 1000 : number) : null;
    }
    const number = Date.parse(value); return Number.isFinite(number) ? number : null;
  }
  function date(value, full = false) {
    const stamp = ms(value); if (stamp === null) return t("Chưa ghi");
    return new Date(stamp).toLocaleString(i18n.dateLocale, full ? { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 } : { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  }
  function duration(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const n = Number(value); if (n < 1000) return `${Math.round(n)} ms`;
    if (n < 60000) return `${(n / 1000).toFixed(n < 10000 ? 2 : 1)} s`;
    return `${Math.floor(n / 60000)}m ${Math.floor((n % 60000) / 1000)}s`;
  }
  function spanDuration(span) {
    if (span.duration_ms !== null && span.duration_ms !== undefined) return Number(span.duration_ms);
    const start = ms(span.started_at), end = ms(span.ended_at);
    return start !== null && end !== null ? Math.max(0, end - start) : null;
  }
  function elapsed(span) {
    const recorded = spanDuration(span); if (recorded !== null) return duration(recorded);
    const start = ms(span.started_at);
    return span.status === "running" && start !== null ? `${duration(Math.max(0, Date.now() - start))} ↻` : t("Chưa ghi thời gian");
  }
  function byteSize(size) {
    return Number.isFinite(Number(size)) ? (Number(size) < 1024 ? `${size} B` : Number(size) < 1048576 ? `${(Number(size) / 1024).toFixed(1)} KB` : `${(Number(size) / 1048576).toFixed(1)} MB`) : "—";
  }
  function toast(message) {
    $("#toast").textContent = message; $("#toast").hidden = false;
    clearTimeout(toast.timer); toast.timer = setTimeout(() => { $("#toast").hidden = true; }, 3500);
  }
  let tokenRefresh = null;
  function tokenNeedsRefresh() {
    if (!state.token) return true;
    try {
      const payload = JSON.parse(atob(state.token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
      return typeof payload.exp === "number" && payload.exp * 1000 <= Date.now() + 30000;
    } catch { return false; } // The backend still validates the signature and claims.
  }
  async function refreshLocalToken() {
    if (!state.config?.local_token_enabled) return;
    if (!tokenRefresh) {
      const base = state.base.replace(/\/+$/, ""), previousToken = state.token;
      tokenRefresh = (async () => {
        const response = await fetch(`${base}/api/v1/inspect/token`, {method:"POST", cache:"no-store"});
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.token) throw new Error(data.detail || t("Không lấy được token local mới."));
        if (state.base.replace(/\/+$/, "") !== base || state.token !== previousToken) throw new Error(t("Cấu hình kết nối đã thay đổi; hãy kết nối lại."));
        state.token = data.token;
        sessionStorage.setItem("core-studio-token", state.token);
      })().finally(() => { tokenRefresh = null; });
    }
    await tokenRefresh;
  }
  async function ensureToken() {
    if (state.config?.local_token_enabled && tokenNeedsRefresh()) await refreshLocalToken();
  }
  async function request(path, options = {}, authenticated = true, retryAuth = true) {
    if (authenticated) await ensureToken();
    const usedToken = state.token;
    const headers = new Headers(options.headers || {});
    if (authenticated && state.token) headers.set("Authorization", `Bearer ${state.token}`);
    const response = await fetch(`${state.base.replace(/\/+$/, "")}${path}`, { ...options, headers, cache: "no-store" });
    if (response.status === 401 && authenticated && retryAuth && state.config?.local_token_enabled) {
      await response.body?.cancel();
      if (state.token === usedToken) await refreshLocalToken();
      return request(path, options, authenticated, false);
    }
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try { const result = await response.json(); message = typeof result.detail === "string" ? result.detail : stringify(result.detail || result); } catch { /* Keep HTTP message. */ }
      throw new Error(message);
    }
    return response.json();
  }
  function replace(node, html) {
    if (node.innerHTML === html) return;
    const scroll = { left: node.scrollLeft, top: node.scrollTop };
    const focus = node.contains(document.activeElement) ? document.activeElement : null;
    const attrs = focus ? ["data-span", "data-collapse", "data-event", "data-artifact", "data-run", "data-json-path", "data-view-mode"].map((key) => [key, focus.getAttribute(key)]).filter(([, value]) => value !== null) : [];
    node.querySelectorAll("[data-json-path]").forEach((branch) => state.jsonOpen.set(branch.dataset.jsonPath, branch.open));
    node.innerHTML = html;
    node.querySelectorAll("[data-json-path]").forEach((branch) => {
      if (state.jsonOpen.has(branch.dataset.jsonPath)) branch.open = state.jsonOpen.get(branch.dataset.jsonPath);
    });
    node.scrollLeft = scroll.left; node.scrollTop = scroll.top;
    if (attrs.length) {
      const [key, value] = attrs[0]; Array.from(node.querySelectorAll(`[${key}]`)).find((el) => el.getAttribute(key) === value)?.focus({ preventScroll: true });
    }
  }
  function connection(ok, message) {
    state.connected = ok; $("#connection").className = `connection ${ok ? "online" : "error"}`;
    $("#connection").textContent = message; $("#connection").title = state.base;
  }
  function polling(ok, message) {
    $("#poll-dot").className = ok ? "" : "error";
    $("#poll-label").textContent = message;
  }
  function renderRuns() {
    const search = $("#run-search").value.toLowerCase().trim(), filter = $("#run-filter").value;
    const rows = state.runs.filter((run) => `${run.session_id} ${run.candidate_name || ""} ${run.target_role || ""}`.toLowerCase().includes(search)).filter((run) => filter === "all" || filter === "completed" && complete.has(run.status) || filter === "failed" && failed.has(run.status) || filter === "active" && statusClass(run.status) === "running");
    replace($("#run-list"), rows.length ? rows.map((run) => `<button class="run-item ${state.selected === run.session_id ? "selected" : ""}" data-run="${esc(run.session_id)}" aria-pressed="${state.selected === run.session_id}"><div class="run-item-top"><span class="status-dot ${statusClass(run.status)}"></span><strong>${esc(run.candidate_name || run.session_id)}</strong></div><div class="run-item-role">${esc(run.target_role || statusLabel(run.status))}</div><div class="run-item-meta"><span class="run-item-id" title="${esc(run.session_id)}">${esc(run.session_id)}</span><span>${esc(date(run.created_at))}</span></div><div class="run-item-meta"><span>${esc(statusLabel(run.status))}</span><span>${run.span_count === undefined ? t("Trace chưa ghi") : `${esc(run.span_count)} ${esc(t("Spans đã ghi"))}`}</span></div></button>`).join("") : `<p class="empty-small">${state.runs.length ? t("Không có lượt chạy khớp bộ lọc.") : t("Chưa có lượt chạy được lưu. Mở FE-test để chạy core.")}</p>`);
  }
  async function refreshRuns(append = false) {
    if (state.runsBusy || !state.connected) return;
    state.runsBusy = true; const generation = state.generation, queryRevision = state.runQueryRevision;
    try {
      const offset = append ? state.runs.length : 0;
      const result = await request(`/api/v1/inspect/runs?limit=200&offset=${offset}&q=${encodeURIComponent(state.runQuery)}`); if (generation !== state.generation || queryRevision !== state.runQueryRevision) return;
      if (append) state.runs = [...state.runs, ...(result.runs || []).filter((row) => !state.runs.some((old) => old.session_id === row.session_id))];
      else state.runs = [...(result.runs || []), ...state.runs.slice(200)];
      state.runTotal = result.total ?? state.runs.length; $("#run-count").textContent = `${state.runs.length} / ${state.runTotal}`;
      $("#load-runs").classList.toggle("hidden", state.runs.length >= state.runTotal); renderRuns();
      if (!state.selected && state.runs.length) {
        const params = new URLSearchParams(location.search), requested = params.get("session_id") || params.get("session");
        selectRun(requested || state.runs[0].session_id);
      }
    } catch (error) {
      if (generation !== state.generation) return;
      connection(false, t("Không đọc được core"));
      if (!state.connectTimer) { const delay = Math.min(30000, 2000 * 2 ** state.connectAttempts); state.connectAttempts += 1; state.connectTimer = setTimeout(connect, delay); }
      if (!state.runs.length) replace($("#run-list"), `<p class="error-message">${esc(error.message)}</p>`);
    } finally { state.runsBusy = false; if (queryRevision !== state.runQueryRevision && generation === state.generation) refreshRuns(); }
  }
  const eventKey = (event) => String(event.event_id ?? event.id ?? `seq:${event.sequence}`);
  function indexEvents() { state.eventIndex = (state.detail?.events || []).map((event) => [event, JSON.stringify(event).toLowerCase()]); }
  function ingestDetail(detail) {
    const oldEvents = state.detail?.events || [];
    const all = new Map([...oldEvents, ...(detail.events || [])].map((event) => [event.event_id || event.id || event.sequence, event]));
    detail.events = Array.from(all.values()).sort((a, b) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0));
    state.eventsMore = typeof detail.summary?.event_count === "number" ? detail.summary.event_count > detail.events.length : Boolean(detail.events_truncated);
    state.detail = detail; indexEvents();
    state.instances = detail.trace?.instances || [];
    state.spans = Array.isArray(detail.spans) ? detail.spans : state.spans;
    state.coverage = detail.trace?.coverage || (state.spans.length ? "legacy_partial" : "not_recorded");
    state.storage = detail.trace?.storage || state.config?.storage || null;
    if (detail.trace?.revision !== undefined) state.revision = detail.trace.revision;
    if (!state.selection && state.spans.length) state.selection = { type: "span", id: state.spans.find((span) => !span.parent_span_id)?.span_id || state.spans[0].span_id };
    renderHeading(); renderContent(); renderDetails();
  }
  async function selectRun(id) {
    if (id === state.selected) return;
    state.selected = id; state.generation += 1; state.detail = null; state.spans = []; state.selection = null; state.artifactContent = null;
    state.revision = null; state.collapsed = new Set(); state.zoom = 1; state.timelineZoom = 1; state.dataModes.clear(); state.jsonOpen.clear(); state.traceBusy = false; state.lastDetailFetch = 0; state.instances = []; state.eventsMore = false;
    renderRuns(); $("#empty-workspace").classList.add("hidden"); $("#run-workspace").classList.remove("hidden");
    replace($("#content-pane"), `<p class="empty-small">${esc(t("Đang đọc dữ liệu đã lưu từ core…"))}</p>`);
    $("#detail-panel").innerHTML = `<div class="detail-placeholder"><h3>${esc(t("Đang đọc trace"))}</h3><p>${esc(t("Dữ liệu được tải từ backend."))}</p></div>`;
    const url = new URL(location.href); url.searchParams.delete("session"); url.searchParams.set("session_id", id); history.replaceState(null, "", url);
    const generation = state.generation;
    try {
      const detail = await request(`/api/v1/inspect/runs/${encodeURIComponent(id)}`);
      if (generation !== state.generation) return;
      state.lastDetailFetch = Date.now(); ingestDetail(detail); polling(true, t("Đã đồng bộ · 1,8 giây/lần"));
    } catch (error) {
      if (generation !== state.generation) return;
      replace($("#content-pane"), `<p class="error-message">${esc(t("Không đọc được run"))}: ${esc(error.message)}</p>`); polling(false, t("Lỗi tải dữ liệu"));
    }
  }
  function renderHeading() {
    const detail = state.detail; if (!detail) return;
    const session = detail.session || {}, row = state.runs.find((run) => run.session_id === state.selected) || {};
    const form = session.form_input || session.input || {}, status = session.lifecycle_status || session.status || row.status;
    $("#run-title").textContent = session.candidate_name || form.candidate_name || row.candidate_name || state.selected;
    document.title = `${$("#run-title").textContent} · Core Studio`;
    $("#run-role").textContent = session.target_role || form.target_role || row.target_role || t("Vai trò chưa ghi");
    $("#run-status").className = `badge ${statusClass(status)}`; $("#run-status").textContent = statusLabel(status);
    $("#run-created").textContent = date(session.created_at || row.created_at);
    $("#copy-id").textContent = state.selected; $("#copy-id").title = `${t("Sao chép")}: ${state.selected}`;
    const roots = state.spans.filter((span) => !span.parent_span_id), run = roots.find((span) => span.kind === "run");
    const measures = [[t("Spans đã ghi"), state.spans.length], [t("Agent instances"), new Set(state.spans.filter((span) => span.kind === "agent" && span.instance_id).map((span) => span.instance_id)).size], [t("Run duration"), run ? elapsed(run) : t("Chưa ghi")]];
    $("#run-measures").innerHTML = measures.map(([label, value]) => `<div class="measure"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join("");
    $("#event-count").textContent = detail.summary?.event_count ?? detail.events?.length ?? "—"; $("#artifact-count").textContent = detail.artifacts?.length ?? "—";
    const coverage = typeof state.coverage === "object" ? state.coverage.status || state.coverage.coverage : state.coverage;
    const hasRecordedTrace = ["recorded", "complete"].includes(coverage);
    const message = detail.execution_origin === "legacy_simulation" ? t("Archive simulation từ phiên bản cũ · dữ liệu cũ được giữ nguyên, không phải lượt chạy provider thật hiện tại.") : hasRecordedTrace ? t("Các bước đã được runtime ghi lại. Trace bắt đầu từ lúc instrumentation được bật; bước cũ chưa ghi sẽ không được suy diễn.") : coverage === "legacy_partial" ? t("Trace lịch sử chưa đầy đủ. Chỉ hiển thị các bước và thời gian đã ghi; không suy diễn bước thiếu.") : t("Run này chưa có trace lifecycle. Events và artifacts vẫn có thể inspect; flow không được dựng giả.");
    $("#coverage-bar").className = `coverage-bar ${hasRecordedTrace ? "" : "warning"}`;
    $("#coverage-bar").innerHTML = `<span>${esc(message)}</span><span class="storage-tag">${esc(t("Lưu trữ"))}: ${esc(state.storage || t("chưa xác định"))}</span>`;
  }
  function hierarchy() {
    const byId = new Map(state.spans.filter((s) => s.span_id).map((s) => [s.span_id, s]));
    const children = new Map(), roots = [];
    for (const span of byId.values()) {
      if (span.parent_span_id && byId.has(span.parent_span_id) && span.parent_span_id !== span.span_id) {
        if (!children.has(span.parent_span_id)) children.set(span.parent_span_id, []);
        children.get(span.parent_span_id).push(span);
      } else roots.push(span);
    }
    const compare = (a, b) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0) || (ms(a.started_at || a.scheduled_at) ?? Infinity) - (ms(b.started_at || b.scheduled_at) ?? Infinity);
    roots.sort(compare); children.forEach((items) => items.sort(compare));
    const rows = [], visited = new Set();
    function walk(span, depth) {
      if (visited.has(span.span_id)) return;
      visited.add(span.span_id); rows.push({ span, depth });
      if (!state.collapsed.has(span.span_id)) (children.get(span.span_id) || []).forEach((child) => walk(child, depth + 1));
    }
    roots.forEach((root) => walk(root, 0));
    // Broken/cyclic historical records remain visible, without inventing a parent.
    for (const span of byId.values()) {
      if (!visited.has(span.span_id) && !Array.from(state.collapsed).some((id) => isDescendant(span, id, byId))) walk(span, 0);
    }
    return { rows, children, byId };
  }
  function isDescendant(span, id, byId) {
    const seen = new Set(); let parent = span.parent_span_id;
    while (parent && !seen.has(parent)) { if (parent === id) return true; seen.add(parent); parent = byId.get(parent)?.parent_span_id; }
    return false;
  }
  const short = (text, n) => String(text ?? "").length > n ? `${String(text).slice(0, n - 1)}…` : String(text ?? "");
  function renderFlow() {
    const { rows, children } = hierarchy(); const width = Math.max(640, ...rows.map((row) => row.depth * 295 + 305)), height = Math.max(260, rows.length * 92 + 35);
    const positions = new Map(rows.map((row, i) => [row.span.span_id, { x: row.depth * 295 + 24, y: i * 92 + 24 }]));
    const edges = rows.map(({ span }) => {
      const child = positions.get(span.span_id), parent = positions.get(span.parent_span_id);
      return parent ? `<path class="flow-edge" d="M ${parent.x + 242} ${parent.y + 34} H ${parent.x + 266} V ${child.y + 34} H ${child.x}"/>` : "";
    }).join("");
    const nodes = rows.map(({ span }) => {
      const { x, y } = positions.get(span.span_id), color = colors[span.kind] || colors.operation;
      const count = children.get(span.span_id)?.length || 0, selected = state.selection?.type === "span" && state.selection.id === span.span_id;
      const metadata = [span.agent, span.lifetime_scope].filter(Boolean).join(" · ");
      const title = `${span.name || span.kind || "Span"} · ${statusLabel(span.status)} · ${elapsed(span)}`;
      return `<g class="flow-node ${selected ? "selected" : ""}" style="--node-color:${color};--node-tint:${color}0d" transform="translate(${x} ${y})" data-span="${esc(span.span_id)}" tabindex="0" role="button" aria-label="${esc(title)}" aria-pressed="${selected}"><title>${esc(title)}</title><rect class="node-box" width="242" height="70" rx="6"/><rect x="0" y="10" width="3" height="50" rx="1" fill="${color}"/><text class="node-kind" x="13" y="17" fill="${color}">${esc((span.kind || "span").toUpperCase())}</text><circle cx="221" cy="14" r="3" fill="${statusClass(span.status) === "failed" ? "#d45863" : statusClass(span.status) === "completed" ? "#229576" : statusClass(span.status) === "running" ? "#5878e8" : "#8b96a6"}"/><text class="node-name" x="13" y="36">${esc(short(span.name || span.agent || "Span", 31))}</text><text class="node-meta" x="13" y="55">${esc(short(metadata || statusLabel(span.status), 25))}</text><text class="node-duration" x="226" y="55" text-anchor="end">${esc(short(elapsed(span), 18))}</text></g>${count ? `<g class="collapse-control" transform="translate(${x + 251} ${y + 25})" tabindex="0" role="button" aria-label="${state.collapsed.has(span.span_id) ? t("Mở") : t("Thu gọn")} ${esc(span.name)}" data-collapse="${esc(span.span_id)}"><rect width="22" height="19" rx="4"/><text x="11" y="13" text-anchor="middle">${state.collapsed.has(span.span_id) ? "+" : "−"}</text><title>${count} ${esc(t("bước con"))}</title></g>` : ""}`;
    }).join("");
    return `<div class="flow-canvas"><svg class="flow-svg" xmlns="http://www.w3.org/2000/svg" width="${width * state.zoom}" height="${height * state.zoom}" viewBox="0 0 ${width} ${height}" aria-label="${esc(t("Flow map từ trace runtime"))}">${edges}${nodes}</svg></div>`;
  }
  function renderWaterfall() {
    const { rows, children } = hierarchy();
    const timed = state.spans.map((span) => ({ start: ms(span.started_at), end: ms(span.ended_at), span })).filter((item) => item.start !== null);
    const origin = timed.length ? Math.min(...timed.map((item) => item.start)) : null;
    const extent = timed.length ? Math.max(...timed.map((item) => item.end ?? (item.span.status === "running" ? Date.now() : item.start))) : null;
    const total = origin === null ? null : Math.max(1, extent - origin);
    const nameWidth = Math.min(285, Math.max(180, $("#content-pane").clientWidth * .4));
    const timelineWidth = Math.max(300, $("#content-pane").clientWidth - nameWidth - 1) * state.timelineZoom;
    const ticks = Math.max(4, Math.min(80, Math.floor(timelineWidth / 125)));
    return `<div class="table-wrap" style="--name-width:${nameWidth}px;--timeline-width:${timelineWidth}px" title="${esc(t("Kéo nền để di chuyển ngang"))}"><div class="waterfall-scale"><span>${esc(t("Span / agent hierarchy"))}</span><div class="time-ticks">${total === null ? esc(t("Không có timing đã ghi")) : Array.from({ length: ticks + 1 }, (_, i) => `<span style="left:${i / ticks * 100}%">${duration(total * i / ticks)}</span>`).join("")}</div></div>${rows.map(({ span, depth }) => {
      const start = ms(span.started_at), end = ms(span.ended_at) ?? (span.status === "running" && start !== null ? Date.now() : null);
      const left = start === null || origin === null ? 0 : Math.max(0, (start - origin) / total * 100), width = end === null || start === null ? null : Math.max(0, (end - start) / total * 100);
      const selected = state.selection?.type === "span" && state.selection.id === span.span_id;
      return `<div class="waterfall-row ${selected ? "selected" : ""} status-${statusClass(span.status)}" role="button" tabindex="0" data-span="${esc(span.span_id)}" aria-label="${esc(span.name)} · ${esc(statusLabel(span.status))} · ${esc(elapsed(span))}"><div class="waterfall-name" style="padding-left:${12 + Math.min(depth, 8) * 15}px"><button class="collapse-inline" ${children.has(span.span_id) ? `data-collapse="${esc(span.span_id)}" aria-label="${esc(t("Thu gọn hoặc mở"))} ${esc(span.name)}"` : 'disabled aria-hidden="true" tabindex="-1"'}>${children.has(span.span_id) ? state.collapsed.has(span.span_id) ? "▸" : "▾" : "·"}</button><span class="kind-dot" style="background:${colors[span.kind] || colors.task}"></span><strong title="${esc(span.name)} · ${esc(statusLabel(span.status))}">${esc(span.name || span.kind)}</strong><span class="status-dot ${statusClass(span.status)}" title="${esc(statusLabel(span.status))}"></span></div><div class="waterfall-time" data-timeline-pan>${width === null ? `<div class="no-timing">${esc(t("Chưa ghi start/end"))}</div>` : `<div class="time-bar ${span.status === "running" ? "running" : ""}" title="${esc(elapsed(span))}" style="left:${left}%;width:${width}%;background-color:${failed.has(span.status) ? "#d45863" : colors[span.kind] || colors.task}"></div><span class="time-duration" style="left:${Math.min(96, left + width)}%">${esc(elapsed(span))}</span>`}</div></div>`;
    }).join("")}</div>`;
  }
  function renderContent() {
    if (!state.detail) return;
    const trace = state.tab === "flow" || state.tab === "waterfall";
    $("#trace-tools").classList.toggle("hidden", !trace); $("#legend").classList.toggle("hidden", !trace);
    $("#zoom-out").disabled = state.tab === "waterfall" && state.timelineZoom <= 1; $("#zoom-in").disabled = state.tab === "waterfall" && state.timelineZoom >= 64; $("#zoom-reset").disabled = false;
    $("#timeline-zoom").classList.toggle("hidden", state.tab !== "waterfall"); $("#timeline-zoom").value = Math.log2(state.timelineZoom);
    $("#zoom-label").textContent = `${Math.round((state.tab === "waterfall" ? state.timelineZoom : state.zoom) * 100)}%`;
    $("#view-caption").textContent = state.tab === "flow" ? t("Quan hệ parent → child được ghi bởi runtime") : t("Timing thực tế · agent và lời gọi theo hierarchy");
    if (trace) {
      replace($("#content-pane"), state.spans.length ? state.tab === "flow" ? renderFlow() : renderWaterfall() : `<div class="trace-empty"><h3>${esc(t("Chưa có trace lifecycle cho run này"))}</h3><p>${esc(t("Core chưa ghi các span có quan hệ parent/child ở lượt chạy này. Chọn Events hoặc Artifacts để xem dữ liệu còn lưu."))}</p><p>${esc(t("Flow và lifetime sẽ xuất hiện khi có trace được ghi từ một lượt chạy mới."))}</p></div>`);
    } else if (state.tab === "events") {
      if (!$("#event-list")) $("#content-pane").innerHTML = `<div class="event-toolbar"><input type="search" id="event-search" placeholder="${esc(t("Lọc nội dung events đã tải"))}" value="${esc(state.eventSearch)}"><small id="event-visible"></small></div><div id="event-list"></div><button id="load-events" class="text-button">${esc(t("Tải thêm events từ backend"))}</button>`;
      renderEvents();
    } else {
      const items = state.detail.artifacts || [];
      replace($("#content-pane"), items.length ? items.map((item) => `<button class="artifact-item ${state.selection?.type === "artifact" && state.selection.id === item.path ? "selected" : ""}" data-artifact="${esc(item.path)}"><span aria-hidden="true">▤</span><strong>${esc(item.path)}</strong><small>${esc(byteSize(item.size_bytes))}</small></button>`).join("") : `<p class="empty-small">${esc(t("Chưa có artifact được lưu cho run này."))}</p>`);
    }
  }
  function renderEvents() {
    const query = state.eventSearch.toLowerCase();
    const events = state.eventIndex.filter(([event, text]) => event.type !== "reasoning_summary" && text.includes(query)).map(([event]) => ({ event, key: eventKey(event) }));
    $("#event-visible").textContent = `${events.length} / ${state.detail.events?.length || 0} ${t("đã tải")}`;
    $("#load-events").classList.toggle("hidden", !state.eventsMore); $("#load-events").disabled = state.eventsBusy;
    replace($("#event-list"), events.length ? events.map(({ event, key }) => `<button class="event-item ${state.selection?.type === "event" && state.selection.id === key ? "selected" : ""}" data-event="${esc(key)}"><span class="event-seq">${esc(event.sequence ?? event.seq ?? "—")}</span><div class="event-main"><b>${esc(event.type || event.name || "Event")}</b><small>${esc(short(stringify(event.payload ?? event.data ?? event), 190))}</small></div><time class="event-time">${esc(date(event.timestamp || event.created_at))}</time></button>`).join("") : `<p class="empty-small">${esc(t("Không có event khớp bộ lọc."))}</p>`);
  }
  function structuredView(value, key, format = "json") {
    const mode = state.dataModes.get(key) || "read", source = stringify(value) ?? "";
    const canRead = format === "markdown" || typeof value === "object" && value !== null;
    const toolbar = `<div class="viewer-toolbar" role="group" aria-label="${esc(t("Nội dung đã lưu"))}"><button class="viewer-tab ${mode === "read" ? "active" : ""}" data-view-mode="read" data-view-key="${esc(key)}" aria-pressed="${mode === "read"}">${esc(t("Xem đẹp"))}</button><button class="viewer-tab ${mode === "source" ? "active" : ""}" data-view-mode="source" data-view-key="${esc(key)}" aria-pressed="${mode === "source"}">${esc(t("Nguồn"))}</button>${format === "json" && mode === "read" && canRead ? `<button class="text-button" data-tree-action="expand" data-view-key="${esc(key)}">${esc(t("Mở cây"))}</button><button class="text-button" data-tree-action="collapse" data-view-key="${esc(key)}">${esc(t("Thu cây"))}</button>` : ""}</div>`;
    const content = mode === "source" || !canRead ? `<pre class="source-view">${esc(source)}</pre>` : format === "markdown" ? viewers.markdown(source) : viewers.json(value, key);
    return `<div class="data-viewer" data-viewer="${esc(key)}">${toolbar}${content}</div>`;
  }
  function dataSection(label, value, extra = "") {
    const present = value !== undefined && value !== null, key = `${state.selection?.id}:${label}`;
    return `<section class="detail-section ${extra}"><h4>${esc(label)}</h4>${present ? structuredView(value, key) : `<p class="data-missing">${esc(t("Không có dữ liệu được ghi cho mục này."))}</p>`}</section>`;
  }
  function facts(items) { return `<dl class="facts">${items.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value ?? t("Chưa ghi"))}</dd>`).join("")}</dl>`; }
  function renderDetails() {
    const selection = state.selection; if (!selection) return;
    if (selection.type === "span") {
      const span = state.spans.find((item) => item.span_id === selection.id); if (!span) return;
      const parent = state.spans.find((item) => item.span_id === span.parent_span_id);
      const instance = state.instances.find((item) => item.instance_id === span.instance_id);
      const start = ms(span.started_at), scheduled = ms(span.scheduled_at), queueTime = start !== null && scheduled !== null ? duration(Math.max(0, start - scheduled)) : t("Chưa ghi");
      const scopeLabels = { application: t("Application · dùng chung toàn tiến trình"), session: t("Session · tồn tại trong phiên"), delegated_call: t("Delegated call · được tạo cho lời gọi") };
      replace($("#detail-panel"), `<div class="detail-head" style="border-top-color:${colors[span.kind] || colors.task}"><div class="detail-kicker"><span style="color:${colors[span.kind] || colors.task}">${esc((span.kind || "span").toUpperCase())}</span><span class="badge ${statusClass(span.status)}">${esc(statusLabel(span.status))}</span></div><h3>${esc(span.name || span.agent || "Span")}</h3><div class="detail-id">${esc(span.span_id)}</div></div><section class="detail-section"><h4>${esc(t("Call lifecycle"))}</h4>${facts([[t("Agent"), span.agent], [t("Instance ID"), span.instance_id], [t("Lifetime scope"), scopeLabels[span.lifetime_scope] || span.lifetime_scope], [t("Parent"), parent?.name || (span.parent_span_id ? `${t("Parent chưa tải")}: ${span.parent_span_id}` : t("Root · không có parent"))], [t("Scheduled"), date(span.scheduled_at, true)], [t("Started"), date(span.started_at, true)], [t("Ended"), date(span.ended_at, true)], [t("Duration"), elapsed(span)], [t("Queue time"), queueTime], [t("Sequence"), span.sequence]])}</section>${instance ? `<section class="detail-section"><h4>${esc(t("Agent instance · observation & release"))}</h4>${facts([[t("Registry status"), instance.status], [t("Observed"), date(instance.observed_at, true)], [t("Released"), instance.end_time_known === false ? t("Không biết thời điểm · process restart") : date(instance.ended_at, true)], [t("Release reason"), instance.end_reason], [t("Scope"), scopeLabels[instance.lifetime_scope] || instance.lifetime_scope], [t("Process ID"), instance.process_id]])}<p class="data-missing">${esc(t("Observed là thời điểm runtime ghi nhận instance. Released là lúc registry trả instance; không phải thời điểm constructor hay destructor."))}</p></section>` : ""}${span.error ? dataSection(t("Error"), span.error, "error") : ""}${dataSection(t("Input"), span.inputs)}${dataSection(t("Output"), span.outputs)}${dataSection(t("Attributes"), span.attributes)}<section class="detail-section"><p class="data-missing">${esc(t("Timing lấy từ timestamp của backend. Ký hiệu ↻ là thời gian đang chạy tính từ started_at đã ghi. Task chỉ có scheduled_at đang chờ chạy; UI không tạo started_at."))}</p></section>`);
    } else if (selection.type === "event") {
      const event = state.detail?.events?.find((item) => eventKey(item) === selection.id); if (!event) return;
      replace($("#detail-panel"), `<div class="detail-head"><div class="detail-kicker">${esc(t("EVENT"))}</div><h3>${esc(event.type || "Event")}</h3><div class="detail-id">${esc(event.id || event.event_id || "")}</div></div><section class="detail-section">${facts([[t("Sequence"), event.sequence ?? event.seq], [t("Timestamp"), date(event.timestamp || event.created_at, true)]])}</section>${dataSection(t("Event payload"), event)}`);
    } else if (selection.type === "artifact") {
      const artifact = state.detail.artifacts?.find((item) => item.path === selection.id);
      replace($("#detail-panel"), `<div class="detail-head"><div class="detail-kicker">${esc(t("PERSISTED ARTIFACT"))}</div><h3>${esc(selection.id)}</h3></div><section class="detail-section">${facts([[t("Size"), artifact ? byteSize(artifact.size_bytes) : "—"], [t("Modified"), date(artifact?.modified_at, true)]])}</section><div class="artifact-preview">${state.artifactContent === null ? `<p class="data-missing">${esc(t("Đang đọc file từ backend…"))}</p>` : structuredView(state.artifactContent, `artifact:${selection.id}`, /\.(md|markdown)$/i.test(selection.id) ? "markdown" : "json")}</div>`);
    }
  }
  async function pollTrace() {
    if (!state.selected || !state.connected || state.traceBusy || !state.detail || document.hidden) return;
    state.traceBusy = true; const id = state.selected, generation = state.generation;
    try {
      const result = await request(`/api/v1/inspect/runs/${encodeURIComponent(id)}/trace`);
      if (generation !== state.generation) return;
      state.spans = Array.isArray(result.spans) ? result.spans : []; state.coverage = result.coverage || "not_recorded"; state.storage = result.storage || state.storage; state.revision = result.revision;
      state.instances = result.instances || state.instances;
      if (!state.selection && state.spans.length) state.selection = { type: "span", id: state.spans.find((span) => !span.parent_span_id)?.span_id || state.spans[0].span_id };
      if (result.events) {
        const merged = new Map([...(state.detail.events || []), ...result.events].map((event) => [event.event_id || event.id || event.sequence, event]));
        state.detail.events = Array.from(merged.values()).sort((a, b) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0)); indexEvents();
      }
      renderHeading(); renderContent(); renderDetails(); polling(true, `${t("Đã đồng bộ")} ${new Date().toLocaleTimeString(i18n.dateLocale)}`);
      if (Date.now() - state.lastDetailFetch > 7000) {
        const detail = await request(`/api/v1/inspect/runs/${encodeURIComponent(id)}`);
        if (generation !== state.generation) return;
        state.lastDetailFetch = Date.now(); ingestDetail(detail);
      }
    } catch (error) {
      if (generation !== state.generation) return;
      polling(false, t("Không đồng bộ được")); $("#poll-label").title = error.message;
    } finally { if (generation === state.generation) state.traceBusy = false; }
  }
  async function connect() {
    clearTimeout(state.connectTimer); state.connectTimer = null;
    const generation = ++state.generation; state.config = null; state.connected = false; state.selected = null; state.detail = null; state.spans = []; state.selection = null; state.traceBusy = false; state.runs = []; state.runsBusy = false;
    $("#run-workspace").classList.add("hidden"); $("#empty-workspace").classList.remove("hidden");
    $("#connection").textContent = t("Đang kết nối core…"); $("#connection").className = "connection";
    try {
      const config = await request("/api/v1/inspect/config", {}, false); if (generation !== state.generation) return;
      state.config = config;
      await ensureToken(); if (generation !== state.generation) return;
      $("#storage-label").textContent = `${t("Lưu trữ")}: ${config.storage || t("chưa xác định")}`;
      $("#provider-label").textContent = `${config.provider || t("Provider chưa cấu hình")}${config.provider_model ? ` · ${config.provider_model}` : ""}${config.real_provider === false ? ` · ${t("không phải provider thật")}` : ""}`;
      connection(true, state.base === location.origin ? t("Core cùng địa chỉ") : new URL(state.base).host);
      state.connectAttempts = 0;
      await refreshRuns();
    } catch (error) {
      if (generation !== state.generation) return;
      connection(false, t("Không kết nối được")); replace($("#run-list"), `<p class="error-message">${esc(error.message)}</p><p class="empty-small">${esc(t("Mở cài đặt nâng cao nếu backend yêu cầu service token."))}</p>`);
      // Keep retrying with backoff so a restarted core is picked up without a manual reload.
      const delay = Math.min(30000, 2000 * 2 ** state.connectAttempts); state.connectAttempts += 1;
      state.connectTimer = setTimeout(connect, delay);
    }
  }
  $("#run-list").addEventListener("click", (event) => { const button = event.target.closest("[data-run]"); if (button) selectRun(button.dataset.run); });
  let searchTimer;
  $("#run-search").addEventListener("input", () => { renderRuns(); clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.runQuery = $("#run-search").value.trim(); state.runQueryRevision++; state.runs = []; refreshRuns(); }, 300); }); $("#run-filter").addEventListener("change", renderRuns);
  $("#load-runs").addEventListener("click", () => refreshRuns(true));
  $("#refresh").addEventListener("click", () => state.connected ? refreshRuns() : connect());
  $(".tabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab]"); if (!button) return;
    state.tab = button.dataset.tab; $("#content-pane").innerHTML = "";
    document.querySelectorAll("[data-tab]").forEach((tab) => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", String(tab === button)); });
    renderContent(); renderDetails();
  });
  $("#content-pane").addEventListener("input", (event) => { if (event.target.id === "event-search") { state.eventSearch = event.target.value; renderEvents(); } });
  async function contentAction(event) {
    if (event.target.closest("#load-events")) {
      if (state.eventsBusy) return;
      state.eventsBusy = true; const generation = state.generation, id = state.selected;
      const after = Math.max(0, ...(state.detail.events || []).map((item) => Number(item.sequence) || 0));
      try {
        const result = await request(`/api/v1/inspect/runs/${encodeURIComponent(id)}/events?after_sequence=${after}&limit=1000`);
        if (generation !== state.generation) return;
        const merged = new Map([...(state.detail.events || []), ...(result.events || [])].map((item) => [item.event_id || item.id || item.sequence, item]));
        state.detail.events = Array.from(merged.values()).sort((a, b) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0)); indexEvents(); state.eventsMore = result.has_more;
        renderEvents();
      } catch (error) { if (generation === state.generation) toast(`${t("Không tải được events")}: ${error.message}`); }
      finally { state.eventsBusy = false; if (state.tab === "events" && $("#event-list")) renderEvents(); }
      return;
    }
    const collapse = event.target.closest("[data-collapse]");
    if (collapse) { state.collapsed.has(collapse.dataset.collapse) ? state.collapsed.delete(collapse.dataset.collapse) : state.collapsed.add(collapse.dataset.collapse); renderContent(); return; }
    const span = event.target.closest("[data-span]"), item = event.target.closest("[data-event]"), artifact = event.target.closest("[data-artifact]");
    if (span) state.selection = { type: "span", id: span.dataset.span };
    else if (item) state.selection = { type: "event", id: item.dataset.event };
    else if (artifact) {
      state.selection = { type: "artifact", id: artifact.dataset.artifact }; state.artifactContent = null;
      const id = state.selected, path = artifact.dataset.artifact, generation = state.generation, requestId = ++state.artifactRequest;
      renderContent(); renderDetails();
      try {
        const result = await request(`/api/v1/inspect/runs/${encodeURIComponent(id)}/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`);
        if (generation !== state.generation || requestId !== state.artifactRequest || state.selection?.type !== "artifact" || state.selection.id !== path) return;
        state.artifactContent = result.content; renderDetails();
      } catch (error) { if (generation === state.generation && requestId === state.artifactRequest) { state.artifactContent = `${t("Không đọc được artifact")}: ${error.message}`; renderDetails(); } }
      return;
    } else return;
    renderContent(); renderDetails();
  }
  $("#content-pane").addEventListener("click", contentAction);
  $("#content-pane").addEventListener("keydown", (event) => { if ((event.key === "Enter" || event.key === " ") && event.target.matches('[role="button"]')) { event.preventDefault(); contentAction(event); } });
  $("#expand-all").addEventListener("click", () => { state.collapsed.clear(); renderContent(); });
  function zoom(value) { state.zoom = Math.min(1.6, Math.max(.35, value)); renderContent(); }
  function zoomTimeline(value) {
    const pane = $("#content-pane"), previous = state.timelineZoom, left = pane.scrollLeft;
    state.timelineZoom = Math.min(64, Math.max(1, value)); renderContent();
    pane.scrollLeft = left * state.timelineZoom / previous;
  }
  $("#timeline-zoom").addEventListener("input", (event) => zoomTimeline(2 ** Number(event.target.value)));
  $("#zoom-in").addEventListener("click", () => state.tab === "waterfall" ? zoomTimeline(state.timelineZoom * 2) : zoom(state.zoom + .15));
  $("#zoom-out").addEventListener("click", () => state.tab === "waterfall" ? zoomTimeline(state.timelineZoom / 2) : zoom(state.zoom - .15));
  $("#zoom-reset").addEventListener("click", () => {
    if (state.tab === "waterfall") { zoomTimeline(1); $("#content-pane").scrollLeft = 0; return; }
    const { rows } = hierarchy(); const width = Math.max(640, ...rows.map((row) => row.depth * 295 + 305));
    zoom(Math.min(1, ($( "#content-pane").clientWidth - 25) / width)); $("#content-pane").scrollTo(0, 0);
  });
  let drag = null, suppressClick = false;
  $("#content-pane").addEventListener("pointerdown", (event) => {
    const waterfall = state.tab === "waterfall" && event.target.closest("[data-timeline-pan], .time-ticks");
    if ((!waterfall && (state.tab !== "flow" || event.target.closest("[data-span], [data-collapse]"))) || event.button !== 0 || event.target.closest("button")) return;
    const pane = $("#content-pane"); drag = { x: event.clientX, y: event.clientY, left: pane.scrollLeft, top: pane.scrollTop, id: event.pointerId, waterfall, moved: false }; suppressClick = false;
  });
  $("#content-pane").addEventListener("pointermove", (event) => {
    if (!drag) return;
    const pane = $("#content-pane"), moved = Math.abs(event.clientX - drag.x) + Math.abs(event.clientY - drag.y) > 5;
    if (!drag.moved && moved) { drag.moved = true; pane.setPointerCapture(drag.id); pane.classList.add("dragging"); }
    if (drag.moved) { pane.scrollLeft = drag.left - event.clientX + drag.x; if (!drag.waterfall) pane.scrollTop = drag.top - event.clientY + drag.y; }
  });
  function releaseDrag() { if (drag) { const pane = $("#content-pane"); suppressClick = drag.moved; if (pane.hasPointerCapture(drag.id)) pane.releasePointerCapture(drag.id); pane.classList.remove("dragging"); drag = null; setTimeout(() => { suppressClick = false; }, 0); } }
  $("#content-pane").addEventListener("click", (event) => { if (suppressClick) { event.preventDefault(); event.stopImmediatePropagation(); } }, true);
  $("#content-pane").addEventListener("pointerup", releaseDrag); $("#content-pane").addEventListener("pointercancel", releaseDrag);
  $("#detail-panel").addEventListener("click", (event) => {
    const mode = event.target.closest("[data-view-mode]"), tree = event.target.closest("[data-tree-action]");
    if (mode) { state.dataModes.set(mode.dataset.viewKey, mode.dataset.viewMode); renderDetails(); }
    if (tree) {
      const viewer = tree.closest("[data-viewer]");
      viewer.querySelectorAll("[data-json-path]").forEach((branch) => { branch.open = tree.dataset.treeAction === "expand"; state.jsonOpen.set(branch.dataset.jsonPath, branch.open); });
    }
  });
  $("#detail-panel").addEventListener("toggle", (event) => { if (event.target.matches("[data-json-path]")) state.jsonOpen.set(event.target.dataset.jsonPath, event.target.open); }, true);
  function setLocale(value) {
    i18n.set(value); renderRuns(); renderHeading(); renderContent(); renderDetails();
    if (state.connected) connection(true, state.base === location.origin ? t("Core cùng địa chỉ") : new URL(state.base).host);
    if (state.config) {
      $("#storage-label").textContent = `${t("Lưu trữ")}: ${state.config.storage || t("chưa xác định")}`;
      $("#provider-label").textContent = `${state.config.provider || t("Provider chưa cấu hình")}${state.config.provider_model ? ` · ${state.config.provider_model}` : ""}${state.config.real_provider === false ? ` · ${t("không phải provider thật")}` : ""}`;
    }
    if (state.detail) polling(true, t("Đã đồng bộ · 1,8 giây/lần"));
    const input = $("#event-search"); if (input) input.placeholder = t("Lọc nội dung events đã tải");
    const load = $("#load-events"); if (load) load.textContent = t("Tải thêm events từ backend");
  }
  $("#ui-language").addEventListener("change", (event) => setLocale(event.target.value));
  $("#copy-id").addEventListener("click", async () => { try { await navigator.clipboard.writeText(state.selected || ""); toast(t("Đã sao chép session ID")); } catch { toast(t("Trình duyệt không cho phép sao chép")); } });
  $("#settings-button").addEventListener("click", () => { $("#base-url").value = state.base; $("#service-token").value = state.token; $("#settings-dialog").showModal(); });
  $("#settings-close").addEventListener("click", () => $("#settings-dialog").close());
  function saveSettings(base, token) {
    state.base = base.replace(/\/+$/, ""); state.token = token; localStorage.setItem("core-studio-base-url", state.base);
    if (token) sessionStorage.setItem("core-studio-token", token); else sessionStorage.removeItem("core-studio-token");
    $("#settings-dialog").close(); connect();
  }
  $("#settings-form").addEventListener("submit", (event) => { event.preventDefault(); const base = $("#base-url").value.trim(); let parsed; try { parsed = new URL(base); } catch { return toast(t("Backend phải dùng http hoặc https")); } if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) return toast(t("Backend phải dùng http hoặc https")); saveSettings(base, $("#service-token").value.trim()); });
  $("#use-local").addEventListener("click", () => saveSettings(location.origin, ""));
  window.addEventListener("storage", (event) => {
    if (event.key === "core-studio-locale") setLocale(event.newValue);
    if (event.key === "core-studio-base-url" && event.newValue && event.newValue !== state.base) { state.base = event.newValue; state.token = sessionStorage.getItem("core-studio-token") || ""; connect(); }
  });
  $("#export-run").addEventListener("click", async () => {
    const id = state.selected; if (!id) return;
    try {
      const data = await request(`/api/v1/inspect/runs/${encodeURIComponent(id)}/export`);
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const link = document.createElement("a"); link.href = url; link.download = `run_${id}.json`;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { toast(`${t("Không xuất được run")}: ${error.message}`); }
  });
  // Arrow keys move the selection through the visible flow/waterfall rows.
  $("#content-pane").addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp"].includes(event.key) || !["flow", "waterfall"].includes(state.tab) || event.target.matches("input, textarea, select")) return;
    const { rows } = hierarchy(); if (!rows.length) return;
    const current = rows.findIndex(({ span }) => state.selection?.type === "span" && state.selection.id === span.span_id);
    const next = rows[Math.max(0, Math.min(rows.length - 1, current + (event.key === "ArrowDown" ? 1 : -1)))];
    event.preventDefault(); state.selection = { type: "span", id: next.span.span_id }; renderContent(); renderDetails();
    const node = Array.from($("#content-pane").querySelectorAll("[data-span]")).find((el) => el.dataset.span === next.span.span_id);
    node?.focus({ preventScroll: true }); node?.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) pollTrace(); });
  setInterval(pollTrace, 1800); setInterval(() => { if (!document.hidden && state.connected) refreshRuns(); }, 7000);
  connect();
})();
