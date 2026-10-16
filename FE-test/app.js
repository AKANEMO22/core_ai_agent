/* Core Studio test client. Runs and transcripts are always read from the backend. */
"use strict";
(async () => {
  // Recover when an older cached HTML page does not include the locale helper yet.
  if (!window.CoreTestI18n) await import("./i18n.js?v=20261014-ux3");
  const $ = (id) => document.getElementById(id);
  const i18n = window.CoreTestI18n;
  const translatedCopy = new Map();
  const t = (source, values) => {
    const text = i18n.t(source, values);
    if (translatedCopy.size > 500) translatedCopy.clear();
    translatedCopy.set(text, {source, values});
    return text;
  };
  const statusLabels = {INITIALIZED:"Đã khởi tạo",PROFILING:"Đang chuẩn bị hồ sơ",READY:"Sẵn sàng",STAGE_ACTIVE:"Đang phỏng vấn",STAGE_TRANSITIONING:"Đang chuyển giai đoạn",DIALOGUE_COMPLETED:"Đã xong hội thoại",ASSESSING:"Đang đánh giá",SYNTHESIZING:"Đang tổng hợp",ENDING:"Đang kết thúc và tạo báo cáo",REPORT_READY:"Báo cáo sẵn sàng",CANCELLED:"Đã hủy",FAILED:"Thất bại",ASSESSMENT_FAILED:"Đánh giá thất bại",EXPIRED:"Đã hết hạn",ARCHIVED:"Đã lưu trữ",UNKNOWN:"Không rõ"};
  const statusText = (value) => t(statusLabels[value] || value || "Không rõ");
  const TERMINAL = new Set(["REPORT_READY", "CANCELLED", "FAILED", "EXPIRED", "ASSESSMENT_FAILED", "ARCHIVED"]);
  const state = {
    base: (localStorage.getItem("core-studio-base-url") || location.origin).replace(/\/$/, ""),
    token: sessionStorage.getItem("core-studio-token") || "", config: null, runs: [],
    id: null, session: null, detail: null, brief: null, artifacts: new Map(), messages: [],
    ws: null, authenticated: false, sequence: 0, events: new Map(), epoch: 0,
    reconnectTimer: null, pollTimer: null, starting: false, busy: false, refreshing: false, creating: false,
    ending: new Map(), reportOpening: null, pending: null, pendingRetryCount: 0, pendingRetryTimer: null,
    connected: false, sourceWarning: "", profile: null, candidateProfile: null,
    runTotal: 0, runQuery: "", runLoading: false, searchTimer: null, runSearchEpoch: 0,
    report: null, autoStart: null, connectTimer: null, connectAttempts: 0, reconnectAttempts: 0, busyTimer: null,
    planSyncKey: null, conversationHtml: "", answerEnabled: false, confirming: new Map(), lastHydrate: 0,
  };
  const uuid = () => crypto.randomUUID();
  const sha256 = async (value) => {
    const bytes = typeof value === "string" ? new TextEncoder().encode(value) : await value.arrayBuffer();
    return [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((byte) => byte.toString(16).padStart(2,"0")).join("");
  };
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const key = (name, id = state.id) => `core-test:${state.base}:${id || "new"}:${name}`;
  localStorage.removeItem(key("pending-create", "new")); // Older builds stored CV text in localStorage.
  const readLocal = (name, id) => { try { return JSON.parse(localStorage.getItem(key(name, id)) || "null"); } catch { return null; } };
  const time = (value) => {
    if (value == null) return "—";
    const d = new Date(typeof value === "number" ? value * 1000 : value);
    return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString(i18n.locale === "vi" ? "vi-VN" : "en-US", {day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
  };
  const shortTime = (value) => {
    if (value == null) return "";
    const d = new Date(typeof value === "number" ? value * 1000 : value);
    return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString(i18n.locale === "vi" ? "vi-VN" : "en-US", {hour:"2-digit",minute:"2-digit"});
  };
  const badgeClass = (status) => ["STAGE_ACTIVE", "REPORT_READY"].includes(status) ? "active" : ["FAILED", "ASSESSMENT_FAILED", "CANCELLED"].includes(status) ? "error" : ["READY", "PROFILING", "STAGE_TRANSITIONING", "DIALOGUE_COMPLETED", "ASSESSING", "SYNTHESIZING", "ENDING"].includes(status) ? "ready" : "";
  function notice(message, isError = false) {
    $("notice").hidden = !message;
    $("notice").textContent = message || "";
    $("notice").className = `notice${isError ? " error" : ""}`;
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
      const base = state.base, previousToken = state.token;
      tokenRefresh = (async () => {
        const response = await fetch(`${base}/api/v1/inspect/token`, {method:"POST", cache:"no-store"});
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.token) throw new Error(data.detail || t("Không lấy được token local mới."));
        if (state.base !== base || state.token !== previousToken) throw new Error(t("Cấu hình kết nối đã thay đổi; hãy kết nối lại."));
        state.token = data.token;
        sessionStorage.setItem("core-studio-token", state.token);
      })().finally(() => { tokenRefresh = null; });
    }
    await tokenRefresh;
  }
  async function ensureToken() {
    if (state.config?.local_token_enabled && tokenNeedsRefresh()) await refreshLocalToken();
  }
  async function api(path, options = {}, authenticated = true, retryAuth = true) {
    if (authenticated) await ensureToken();
    const usedToken = state.token;
    const headers = new Headers(options.headers || {});
    if (authenticated && state.token) headers.set("Authorization", `Bearer ${state.token}`);
    if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const response = await fetch(`${state.base}${path}`, {...options, headers, cache: "no-store"});
    if (response.status === 401 && authenticated && retryAuth && state.config?.local_token_enabled) {
      await response.body?.cancel();
      if (state.token === usedToken) await refreshLocalToken();
      return api(path, options, authenticated, false);
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : data.detail ? JSON.stringify(data.detail) : response.statusText;
      const error = new Error(`${response.status}: ${detail}`);
      error.status = response.status;
      throw error;
    }
    return data;
  }
  function connectionLabel() {
    const connected = state.connected;
    $("connection-state").className = `connection-chip${connected ? " connected" : ""}`;
    $("connection-state").textContent = connected ? (state.id && state.authenticated ? t("Core + stream đang kết nối") : t("Core đã kết nối")) : t("Core chưa kết nối");
  }
  function canRun() { return state.connected && state.config?.real_provider === true; }
  function canRunSelected() { return canRun() && state.detail?.execution_origin !== "legacy_simulation"; }
  function renderProvider() {
    $("backend-label").textContent = state.base;
    $("model-label").textContent = state.config?.provider_model || state.config?.provider || t("Chưa xác nhận");
    $("storage-label").textContent = state.config?.storage === "sqlite" ? t("SQLite + workspace trên đĩa") : state.config?.storage === "postgres" ? t("PostgreSQL + workspace trên đĩa") : state.config?.storage || t("Backend chưa xác nhận lưu trữ");
    const real = state.config?.real_provider === true;
    $("provider-tag").textContent = real ? t("PROVIDER THỰC") : t("PROVIDER CHƯA SẴN SÀNG");
    $("provider-tag").className = `real-tag${real ? " available" : ""}`;
    $("provider-warning").hidden = real;
    $("provider-warning-detail").textContent = state.config ? `Provider: ${state.config.provider || t("không xác định")}. ${state.config.provider_error || t("Core phải được cấu hình provider thật; chức năng chạy bị khóa khi provider là mock hoặc chưa sẵn sàng.")}` : t("Không đọc được cấu hình core. Kiểm tra kết nối backend.");
    $("new-session").disabled = !canRun();
    if ($("empty-create")) $("empty-create").disabled = !canRun();
    $("create-submit").disabled = !canRun() || state.creating;
    connectionLabel();
  }
  async function connect() {
    clearTimeout(state.connectTimer);
    state.connectTimer = null;
    disconnectStream();
    state.connected = false;
    state.config = null;
    renderProvider();
    try {
      state.config = await api("/api/v1/inspect/config", {}, false);
      await ensureToken();
      if (!state.token) throw new Error(t("Backend yêu cầu service token. Nhập token trong ‘Kết nối khác’."));
      await refreshRuns(true);
      state.connected = true;
      state.connectAttempts = 0;
      notice("");
      renderProvider();
      const requested = new URL(location.href).searchParams.get("session_id");
      if (requested) await selectRun(requested);
    } catch (error) {
      // Retry automatically so a restarted core is picked up without a manual reload.
      const delay = Math.min(30000, 2000 * 2 ** state.connectAttempts);
      state.connectAttempts += 1;
      notice(t("Không kết nối được core: {error}. Tự thử lại sau {seconds} giây.",{error:error.message,seconds:Math.round(delay / 1000)}), true);
      state.connectTimer = setTimeout(connect, delay);
      $("run-list").innerHTML = `<p class="sidebar-empty">${t("Chưa đọc được lịch sử từ backend.")}</p>`;
      renderProvider();
    }
  }
  async function refreshRuns(throwError = false, append = false) {
    if (state.runLoading && !throwError) return;
    state.runLoading = true;
    const searchEpoch = state.runSearchEpoch;
    const offset = append ? state.runs.length : 0;
    try {
      const data = await api(`/api/v1/inspect/runs?limit=100&offset=${offset}&q=${encodeURIComponent(state.runQuery)}`);
      if (searchEpoch !== state.runSearchEpoch) return;
      const rows = Array.isArray(data.runs) ? data.runs : [];
      state.runs = append ? [...state.runs, ...rows.filter((row) => !state.runs.some((old) => old.session_id === row.session_id))] : [...rows, ...state.runs.slice(100).filter((row) => !rows.some((fresh) => fresh.session_id === row.session_id))];
      state.runTotal = data.total ?? state.runs.length;
      renderRuns();
    } catch (error) {
      if (throwError) throw error;
      notice(t("Không tải được lịch sử: {error}",{error:error.message}), true);
    } finally { state.runLoading = false; }
  }
  function renderRuns() {
    const rows = state.runs;
    $("more-runs").hidden = state.runs.length >= state.runTotal;
    if (!rows.length) {
      $("run-list").innerHTML = `<p class="sidebar-empty">${state.runQuery ? t("Không tìm thấy phiên phù hợp.") : t("Chưa có phiên đã lưu. Tạo phiên bằng hồ sơ của bạn để bắt đầu.")}</p>`;
      return;
    }
    $("run-list").innerHTML = rows.map((run) => `<button type="button" class="run-item${run.session_id === state.id ? " selected" : ""}" data-session="${escape(run.session_id)}"><strong>${escape(run.candidate_name || run.session_id)}</strong><small>${escape(run.target_role || run.session_id)}</small><div class="run-meta"><span class="status-badge ${badgeClass(run.status)}" title="${escape(run.status || "UNKNOWN")}">${escape(statusText(run.status))}</span><span>${escape(time(run.updated_at || run.created_at))}</span></div></button>`).join("");
    $("run-list").querySelectorAll("[data-session]").forEach((button) => button.addEventListener("click", () => selectRun(button.dataset.session).catch((error) => notice(error.message, true))));
  }
  function disconnectStream() {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
    state.authenticated = false;
    const ws = state.ws;
    state.ws = null;
    if (ws) { ws.onclose = null; ws.close(); }
    connectionLabel();
  }
  async function selectRun(id) {
    disconnectStream();
    if (state.pendingRetryTimer) clearTimeout(state.pendingRetryTimer);
    state.pendingRetryTimer = null;
    state.pendingRetryCount = 0;
    state.epoch += 1;
    state.id = id;
    state.detail = null;
    state.session = null;
    state.brief = null;
    state.profile = null;
    state.messages = [];
    state.artifacts.clear();
    state.events.clear();
    state.sequence = 0;
    state.busy = false;
    state.starting = false;
    state.pending = readLocal("pending-answer");
    state.autoStart = null;
    state.sourceWarning = "";
    state.planSyncKey = null;
    state.conversationHtml = "";
    state.reconnectAttempts = 0;
    $("answer").value = localStorage.getItem(key("draft")) || "";
    const url = new URL(location.href);
    url.searchParams.set("session_id", id);
    history.replaceState(null, "", url);
    renderRuns();
    renderSession();
    const epoch = state.epoch;
    await hydrate(epoch);
    if (epoch === state.epoch && state.session && !TERMINAL.has(state.session.status) && canRunSelected()) openStream(epoch);
  }
  async function artifact(path, fingerprint, id, epoch) {
    const cached = state.artifacts.get(path);
    if (cached?.fingerprint === fingerprint) return cached.data;
    const encodedPath = path.split("/").map(encodeURIComponent).join("/");
    const response = await api(`/api/v1/inspect/runs/${encodeURIComponent(id)}/artifacts/${encodedPath}`);
    const data = response.content?.payload || response.content;
    if (epoch === state.epoch) state.artifacts.set(path, {fingerprint, data});
    return data;
  }
  async function hydrate(epoch = state.epoch) {
    if (!state.id) return;
    const id = state.id;
    const detail = await api(`/api/v1/inspect/runs/${encodeURIComponent(id)}`);
    if (epoch !== state.epoch) return;
    state.detail = detail;
    state.session = detail.session;
    state.lastHydrate = Date.now();
    const events = Array.isArray(detail.events) ? detail.events : [];
    events.forEach(ingestEvent);
    const entries = (detail.artifacts || []).filter((a) => /^turns\/turn_\d+_(candidate|interviewer)\.json$/.test(a.path) || ["inputs/form_input.json","context/interview_brief.json","context/candidate_profile.json"].includes(a.path));
    let failedCount = 0;
    const loaded = await Promise.all(entries.map(async (entry) => {
      try { return {path:entry.path, data:await artifact(entry.path, `${entry.size_bytes}:${entry.modified_at}`, id, epoch)}; }
      catch { failedCount += 1; return null; }
    }));
    if (epoch !== state.epoch) return;
    state.messages = loaded.filter(Boolean).filter((entry) => entry.path.startsWith("turns/")).map((entry) => {
      const match = entry.path.match(/turn_(\d+)_(candidate|interviewer)/);
      return {turn:Number(match[1]), role:match[2], text:String(entry.data?.content || ""), timestamp:entry.data?.timestamp, messageId:entry.data?.message_id};
    }).sort((a, b) => a.turn - b.turn || (a.role === "candidate" ? -1 : 1));
    const form = loaded.find((entry) => entry?.path === "inputs/form_input.json")?.data;
    state.profile = form || null;
    state.brief = loaded.find((entry) => entry?.path === "context/interview_brief.json")?.data || null;
    state.candidateProfile = loaded.find((entry) => entry?.path === "context/candidate_profile.json")?.data || null;
    state.sourceWarning = failedCount ? t("{count} artifact chưa đọc được; hội thoại có thể chưa đầy đủ.",{count:failedCount}) : detail.events_truncated ? t("Event replay bị giới hạn; hội thoại được khôi phục từ artifact đã lưu.") : "";
    if (detail.execution_origin === "legacy_simulation") state.sourceWarning = t("Phiên simulation cũ · chỉ xem archive. Tạo phiên mới để chạy provider thật.");
    if (state.pending) {
      const candidate = state.messages.find((m) => m.role === "candidate" && (m.messageId === state.pending.message_id || m.turn === state.pending.expected_turn && m.text === state.pending.candidate_answer));
      if (candidate) { state.pending.accepted = true; localStorage.setItem(key("pending-answer"), JSON.stringify(state.pending)); }
      const completed = state.messages.some((m) => m.role === "interviewer" && m.turn === state.pending.expected_turn);
      if (completed) {
        state.pending = null;
        state.busy = false;
        localStorage.removeItem(key("pending-answer"));
        localStorage.removeItem(key("draft"));
        $("answer").value = "";
      }
    }
    localStorage.removeItem(key("auto-start"));
    state.autoStart = null;
    if (TERMINAL.has(state.session.status) || detail.execution_origin === "legacy_simulation") disconnectStream();
    renderSession();
    if (state.session.status === "REPORT_READY" && state.ending.get(id) === "accepted") await openReport(epoch);
    else if (TERMINAL.has(state.session.status)) state.ending.delete(id);
  }
  function isEnding() { return state.ending.has(state.id) || state.session?.status === "ENDING"; }
  async function openReport(epoch = state.epoch) {
    const id = state.id;
    if (!id || state.reportOpening === id) return;
    state.reportOpening = id;
    try {
      const report = await api(`/api/v1/sessions/${encodeURIComponent(id)}/report`);
      if (epoch !== state.epoch || id !== state.id) return;
      state.report = report;
      renderReport();
      if (!$("report-dialog").open) $("report-dialog").showModal();
      state.ending.delete(id);
      renderSession();
    } finally { if (state.reportOpening === id) state.reportOpening = null; }
  }
  function ingestEvent(event) {
    if (!event || !event.type || event.type === "reasoning_summary") return;
    const eventKey = event.event_id || `${event.sequence}:${event.type}:${event.turn_id}`;
    state.events.set(eventKey, event);
    state.sequence = Math.max(state.sequence, Number(event.sequence) || 0);
  }
  function renderConversation() {
    if (!state.id) return;
    const container = $("conversation");
    const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
    const messages = [...state.messages];
    const savedTurns = new Set(messages.filter((m) => m.role === "interviewer").map((m) => m.turn));
    const liveTurns = new Map();
    [...state.events.values()].sort((a,b) => (a.sequence || 0) - (b.sequence || 0)).forEach((event) => {
      const match = String(event.turn_id || "").match(/(?:turn[_-]?)(\d+)/);
      const turn = event.payload?.turn_number ?? (match ? Number(match[1]) : null);
      if (turn == null || savedTurns.has(turn)) return;
      const live = liveTurns.get(turn) || {turn, role:"interviewer", text:"", timestamp:event.timestamp, live:true};
      if (event.type === "response_token") live.text += event.payload?.token || "";
      if (event.type === "complete") { live.text = event.payload?.final_response || live.text; live.live = false; }
      if (event.type === "turn_complete") { live.text = event.payload?.interviewer_response || live.text; live.live = false; }
      liveTurns.set(turn, live);
    });
    messages.push(...[...liveTurns.values()].filter((m) => m.text));
    messages.sort((a,b) => a.turn - b.turn || (a.role === "candidate" ? -1 : 1));
    if (!messages.length) {
      const status = state.session?.status;
      const preparing = ["INITIALIZED","PROFILING","READY"].includes(status);
      state.conversationHtml = "";
      container.innerHTML = `<div class="empty-conversation"><div class="empty-symbol">${preparing ? "↻" : "↗"}</div><h3>${preparing ? t("Core đang chuẩn bị phiên") : t("Chưa có hội thoại đã lưu")}</h3><p>${preparing ? t("Hồ sơ, ngữ cảnh và nội dung phỏng vấn đang được xử lý. Câu hỏi đầu tiên sẽ xuất hiện tại đây.") : t("Phiên này chưa có artifact hội thoại. Trạng thái và chi tiết runtime được lưu tại inspector.")}</p></div>`;
      return;
    }
    const html = messages.map((m) => `<article class="message ${m.role}${m.live ? " live" : ""}"><span class="avatar">${m.role === "candidate" ? t("Bạn") : "C"}</span><div class="message-body"><div class="message-header">${m.role === "candidate" ? t("Bạn") : t("Core · Người phỏng vấn")}<small>${t("Lượt")} ${m.turn} ${shortTime(m.timestamp) ? `· ${escape(shortTime(m.timestamp))}` : ""}${m.live ? t(" · đang truyền trực tiếp") : ""}</small></div><div class="message-content">${escape(m.text)}</div></div></article>`).join("");
    // Polling re-renders often; keep the DOM (and any text selection) when nothing changed.
    if (html === state.conversationHtml && container.querySelector(".message")) return;
    state.conversationHtml = html;
    container.innerHTML = html;
    if (atBottom || state.busy) container.scrollTop = container.scrollHeight;
  }
  function renderSession() {
    const s = state.session;
    const status = s?.status || "—";
    const profile = state.profile;
    $("session-title").textContent = profile?.candidate_name || (state.id ? t("Đang đọc phiên…") : t("Chưa chọn phiên chạy"));
    $("session-subtitle").textContent = state.id ? `${profile?.target_role || ""}${profile?.target_role ? " · " : ""}${state.id}` : t("Chọn lịch sử hoặc tạo phiên bằng dữ liệu của bạn.");
    $("session-status").textContent = statusText(status);
    $("session-status").title = status;
    $("session-status").className = `status-badge ${badgeClass(status)}`;
    const fields = [[t("Mã phiên"),state.id], [t("Trạng thái"),statusText(status)], [t("Lượt hiện tại"),s?.current_turn], [t("Giai đoạn"),s?.current_stage_id], [t("Ngôn ngữ phỏng vấn"), profile?.interview_language ? t(profile.interview_language === "vi" ? "Tiếng Việt" : "Tiếng Anh") : t("Chưa được ghi nhận")], [t("Cập nhật"),time(s?.updated_at)], [t("Luồng trực tiếp"),state.authenticated ? t("Đã kết nối") : t("Chưa kết nối")]];
    $("session-info").innerHTML = fields.map(([label,value]) => `<div><dt>${escape(label)}</dt><dd>${escape(value ?? "—")}</dd></div>`).join("");
    const brief = state.brief;
    const planCard = $("plan-card");
    planCard.hidden = !brief;
    if (brief) {
      const roundLabels = {hr:"HR",technical:t("Kỹ thuật"),hiring_manager:t("Quản lý tuyển dụng"),combined:t("Tổng hợp")};
      const duration = brief.duration || {};
      const range = Array.isArray(duration.range_minutes) ? ` (${duration.range_minutes[0]}–${duration.range_minutes[1]})` : "";
      $("plan-info").innerHTML = `<p><strong>${escape(t("Cấp độ"))}:</strong> ${escape(brief.seniority?.value || t("Cần xác nhận"))}</p><p><strong>${escape(t("Vòng"))}:</strong> ${escape(roundLabels[brief.interview_round?.value] || brief.interview_round?.value || "—")}</p><p><strong>${escape(t("Thời lượng"))}:</strong> ${escape(duration.planned_minutes ?? "—")}${escape(range)} ${escape(t("phút"))} · ${escape(duration.source || "—")}</p><p><strong>${escape(t("Trạng thái nguồn"))}:</strong> ${escape(brief.source_snapshot?.status || "unavailable")}</p>`;
      const limitations = Array.isArray(brief.limitations) ? brief.limitations : [];
      $("plan-limitations").textContent = limitations.length ? `${t("Cần xem xét trước khi bắt đầu")}: ${limitations.join(" ")}` : t("Chưa ghi nhận giới hạn nguồn.");
      $("plan-card").querySelector(".plan-sources")?.remove();
      const evidenceIds = new Set([...(brief.interview_round?.evidence_ids || []), ...(duration.evidence_ids || [])]);
      const sources = (brief.source_snapshot?.sources || []).slice(0,8);
      if (sources.length) {
        $("plan-limitations").insertAdjacentHTML("afterend", `<ul class="plan-sources">${sources.map((source) => {
          const url = String(source.url || "");
          const link = /^https:\/\//i.test(url) ? `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(source.title || url)}</a>` : escape(source.title || source.source_id);
          const status = source.verification_status || "unreviewed";
          const excerpt = String(source.content_excerpt || source.snippet || "").slice(0,320);
          const use = evidenceIds.has(source.source_id) ? t("Đã dùng để suy luận") : t("Nguồn tham khảo");
          return `<li>${link} <small>(${escape(status)} · ${escape(use)})</small>${excerpt ? `<p>${escape(excerpt)}</p>` : ""}</li>`;
        }).join("")}</ul>`);
      }
      // Only overwrite the plan form when the saved plan changes, so polling never discards edits in progress.
      const planSyncKey = JSON.stringify([state.id, state.profile?.seniority, state.profile?.interview_round, state.profile?.duration_minutes]);
      if (planSyncKey !== state.planSyncKey) {
        state.planSyncKey = planSyncKey;
        $("plan-seniority").value = state.profile?.seniority || "";
        $("plan-round").value = state.profile?.interview_round || "";
        $("plan-duration").value = state.profile?.duration_minutes ?? "";
      }
      $("plan-duration").placeholder = String(duration.planned_minutes ?? 40);
      $("plan-duration-hint").textContent = `${t("Đề xuất hiện tại")}: ${duration.planned_minutes ?? 40} ${t("phút")} · ${duration.source || "—"}`;
      ["plan-seniority","plan-round","plan-duration","save-plan"].forEach((id) => { $(id).disabled = status !== "READY" || state.starting || isEnding(); });
    }
    $("inspect-run").href = state.id ? `/inspect/?session_id=${encodeURIComponent(state.id)}` : "/inspect/";
    $("start-session").disabled = !canRunSelected() || status !== "READY" || state.starting || isEnding();
    $("start-session").textContent = state.starting ? t("Đang tạo câu hỏi…") : t("Bắt đầu phỏng vấn");
    $("cancel-session").disabled = !s || TERMINAL.has(status) || isEnding();
    if (!state.confirming.has("cancel-session")) $("cancel-session").textContent = t("Hủy phiên");
    $("end-session").disabled = !s || TERMINAL.has(status) || isEnding() || state.detail?.execution_origin === "legacy_simulation";
    if (!state.confirming.has("end-session")) $("end-session").textContent = isEnding() ? t("Đang kết thúc và tạo báo cáo…") : t("Kết thúc và tạo báo cáo");
    const latestAnswerTurn = Math.max(-1,...state.messages.filter((m) => m.role === "candidate").map((m) => m.turn));
    const latestQuestionTurn = Math.max(-1,...state.messages.filter((m) => m.role === "interviewer").map((m) => m.turn));
    const hasQuestion = latestQuestionTurn >= 0 && latestQuestionTurn >= latestAnswerTurn;
    const answerEnabled = canRunSelected() && status === "STAGE_ACTIVE" && state.authenticated && hasQuestion && !state.busy && !state.pending && !isEnding();
    state.answerEnabled = answerEnabled;
    $("answer").disabled = !answerEnabled;
    $("send-answer").disabled = !answerEnabled || !$("answer").value.trim();
    $("answer-hint").textContent = state.sourceWarning || (s?.failure_code ? t("Core thất bại: {code}",{code:s.failure_code}) : state.pending ? t("Đang chờ xác nhận từ dữ liệu đã lưu") : state.busy ? t("Core đang xử lý lượt hiện tại…") : answerEnabled ? t("Ctrl / ⌘ + Enter để gửi · lưu ở backend") : !canRun() ? t("Cần provider thật để chạy") : status === "STAGE_ACTIVE" && !state.authenticated ? t("Đang kết nối lại stream…") : TERMINAL.has(status) ? t("Phiên đã kết thúc · hội thoại từ backend") : t("Chờ core sẵn sàng"));
    $("pending-panel").hidden = !state.pending;
    $("pending-text").textContent = state.pending?.accepted ? t("Core đã lưu câu trả lời. Đang chờ câu hỏi tiếp theo; có thể gửi lại cùng ID để khôi phục lượt chưa hoàn tất.") : t("Đã gửi nhưng chưa xác nhận được từ artifact. Chờ hoặc gửi lại cùng ID; không tạo lượt trả lời mới.");
    $("retry-answer").disabled = !canRunSelected() || !state.authenticated || status !== "STAGE_ACTIVE" || state.busy || isEnding();
    $("show-report").disabled = status !== "REPORT_READY";
    $("download-transcript").disabled = !state.messages.length;
    $("report-hint").textContent = isEnding() && status !== "REPORT_READY" ? t("Core đang dừng các agent và tổng hợp bằng chứng hiện có. Báo cáo có thể chưa đủ dữ liệu.") : status === "REPORT_READY" ? t("Core đã lưu báo cáo hoàn tất. Mở để đọc kết quả.") : t("Báo cáo xuất hiện khi core hoàn tất đánh giá.");
    const extracted = state.candidateProfile || {};
    const entries = (value) => Array.isArray(value) ? value.slice(0,5) : [];
    const renderItems = (items, formatter) => items.length ? `<ul>${items.map((item) => `<li>${formatter(item)}</li>`).join("")}</ul>` : "";
    const extractedSections = [
      [t("Mục tiêu nghề nghiệp"), extracted.career_goals],
      [t("Kinh nghiệm"), entries(extracted.experience).map((item) => `${item.role || ""}${item.company ? ` · ${item.company}` : ""}${item.period ? ` (${item.period})` : ""}${item.responsibilities?.length ? `: ${item.responsibilities.slice(0,2).join("; ")}` : ""}`)],
      [t("Dự án"), entries(extracted.projects).map((item) => `${item.name || t("Dự án")}${item.technologies?.length ? ` · ${item.technologies.join(", ")}` : ""}${item.personal_contribution ? ` — ${item.personal_contribution}` : ""}`)],
      [t("Kỹ năng trích xuất"), entries(extracted.skills).map((item) => typeof item === "string" ? item : item.name)],
      [t("Học vấn"), entries(extracted.education).map((item) => `${item.institution || ""}${item.field ? ` · ${item.field}` : ""}${item.period ? ` (${item.period})` : ""}`)],
      [t("Giải thưởng"), entries(extracted.awards).map((item) => typeof item === "string" ? item : item.name)],
      [t("Chứng chỉ"), entries(extracted.certifications).map((item) => typeof item === "string" ? item : item.name)],
    ].filter(([, items]) => Array.isArray(items) && items.length);
    $("profile-info").innerHTML = profile ? `<p><strong>${escape(profile.candidate_name)}</strong>${profile.target_role ? ` · ${escape(profile.target_role)}` : ""}</p><p>${escape(profile.target_company || "")} · ${escape(profile.experience_years ?? "—")} ${t("năm kinh nghiệm")}</p><p>${escape(extracted.summary || t("Chưa trích xuất được phần tóm tắt."))}</p>${extractedSections.map(([title,items]) => `<div class="profile-extracted"><strong>${escape(title)}</strong>${renderItems(items, (item) => escape(item))}</div>`).join("")}` : t("Chưa đọc được hồ sơ đã lưu.");
    $("edit-profile").disabled = !profile || !state.candidateProfile || status !== "READY" || state.starting || isEnding();
    renderConversation();
    connectionLabel();
  }
  function openStream(epoch = state.epoch) {
    if (!state.id || TERMINAL.has(state.session?.status) || epoch !== state.epoch) return;
    disconnectStream();
    const wsUrl = new URL(`${state.base}/ws/sessions/${encodeURIComponent(state.id)}/stream`);
    wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(wsUrl.href);
    state.ws = ws;
    ws.onopen = async () => {
      try {
        await ensureToken();
        if (epoch === state.epoch && state.ws === ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:"auth", token:state.token}));
      } catch (error) {
        if (epoch === state.epoch && state.ws === ws) { notice(error.message,true); ws.close(1000,"Token refresh failed"); }
      }
    };
    ws.onmessage = (message) => {
      if (epoch !== state.epoch) return;
      let event;
      try { event = JSON.parse(message.data); } catch { return; }
      if (event.type === "auth_ok") {
        state.authenticated = true;
        state.reconnectAttempts = 0;
        ws.send(JSON.stringify({type:"replay", last_sequence:state.sequence}));
        renderSession();
        return;
      }
      if (["pong","auth"].includes(event.type)) return;
      if (event.type === "resync_required") { hydrate(epoch).catch((error) => notice(error.message,true)); return; }
      ingestEvent(event);
      if (["error","busy","generation_cancelled"].includes(event.type)) {
        const code = event.payload?.code;
        const transientBusy = event.type === "busy" || ["TURN_IN_PROGRESS","GENERATION_BUSY"].includes(code);
        if (transientBusy && !isEnding() && state.pending && state.pendingRetryCount < 5) {
          state.pendingRetryCount += 1;
          state.busy = true;
          if (state.pendingRetryTimer) clearTimeout(state.pendingRetryTimer);
          const retryAfter = Number(event.payload?.retry_after_ms) || 250;
          const delay = Math.max(150, Math.min(2000, retryAfter));
          notice(t("Core đang hoàn tất lượt trước; câu trả lời sẽ được gửi lại tự động."));
          state.pendingRetryTimer = setTimeout(() => {
            state.pendingRetryTimer = null;
            sendPending(true);
          }, delay);
          renderSession();
          return;
        }
        state.busy = false;
        notice(t("Core: {error}. Đọc trạng thái và artifact trước khi thử lại.",{error:event.payload?.message || event.payload?.error || event.payload?.code || event.type}), true);
      }
      if (event.type === "turn_complete") {
        state.busy = false;
        state.pendingRetryCount = 0;
        if (state.pendingRetryTimer) clearTimeout(state.pendingRetryTimer);
        state.pendingRetryTimer = null;
      }
      if (["turn_complete","session_failed","session_cancelled","report_ready","stage_transition"].includes(event.type)) hydrate(epoch).catch((error) => notice(error.message,true));
      renderSession();
    };
    ws.onclose = (event) => {
      if (epoch !== state.epoch || state.ws !== ws) return;
      state.ws = null;
      state.authenticated = false;
      state.busy = false;
      renderSession();
      if ([1008,4001,4003,4009].includes(event.code)) { notice(t("Stream bị đóng ({code}): {reason}.",{code:event.code,reason:event.reason || t("kiểm tra token hoặc phiên đang mở ở tab khác")}),true); return; }
      if (!TERMINAL.has(state.session?.status)) {
        const delay = Math.min(15000, 1000 * 2 ** state.reconnectAttempts);
        state.reconnectAttempts += 1;
        state.reconnectTimer = setTimeout(() => openStream(epoch), delay);
      }
    };
    ws.onerror = () => { if (epoch === state.epoch) notice(t("Không kết nối được stream. Lịch sử vẫn được đọc từ backend."),true); };
  }
  async function startInterview() {
    if (!state.id || !canRunSelected() || state.session?.status !== "READY" || state.starting || isEnding()) return;
    const id = state.id;
    const epoch = state.epoch;
    state.starting = true;
    state.busy = true;
    renderSession();
    try {
      await api(`/api/v1/sessions/${encodeURIComponent(id)}/start?confirm_limitations=true`, {method:"POST"});
      if (epoch === state.epoch) {
        notice("");
        state.autoStart = null;
        localStorage.removeItem(key("auto-start"));
        await hydrate(epoch);
      }
    } catch (error) {
      if (epoch === state.epoch) {
        state.autoStart = null;
        localStorage.removeItem(key("auto-start"));
        notice(t("Core chưa bắt đầu được: {error}. Có thể thử nút ‘Bắt đầu phỏng vấn’ khi phiên READY.",{error:error.message}),true);
      }
    } finally {
      if (epoch === state.epoch) { state.starting = false; state.busy = false; renderSession(); }
    }
  }
  function sendPending(automaticRetry = false) {
    if (isEnding() || !state.pending || !canRunSelected() || !state.authenticated || state.ws?.readyState !== WebSocket.OPEN) return;
    if (!automaticRetry) state.pendingRetryCount = 0;
    if (state.pendingRetryTimer) clearTimeout(state.pendingRetryTimer);
    state.pendingRetryTimer = null;
    state.busy = true;
    state.ws.send(JSON.stringify({...state.pending, type:"candidate_answer", last_sequence:state.sequence}));
    renderSession();
    // Do not send again automatically. Only the operator may retry the same message ID.
    clearTimeout(state.busyTimer);
    state.busyTimer = setTimeout(() => {
      state.busyTimer = null;
      if (state.pending && state.busy) { state.busy = false; renderSession(); }
    }, 60000);
  }
  const profileFormFields = ["candidate_name", "target_role", "target_company", "experience_years", "skills"];
  const profileCvFields = ["cv_file", "cv_text"];
  function syncProfileInputMode() {
    const useCv = $("profile-input-mode").value === "cv";
    for (const name of [...profileFormFields, ...profileCvFields]) {
      const field = $("create-form").elements.namedItem(name);
      const active = profileCvFields.includes(name) === useCv;
      field.disabled = !active;
      field.closest("label").hidden = !active;
    }
  }
  $("profile-input-mode").addEventListener("change", syncProfileInputMode);
  syncProfileInputMode();
  $("create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!canRun() || state.creating) return;
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const payload = {
      candidate_name:String(form.get("candidate_name") || "").trim(), target_role:String(form.get("target_role") || "").trim(),
      target_company:String(form.get("target_company") || "").trim(), experience_years:String(form.get("experience_years") || "").trim() ? Number(form.get("experience_years")) : null,
      seniority:String(form.get("seniority") || "").trim() || null,
      interview_round:String(form.get("interview_round") || "").trim() || null,
      duration_minutes:String(form.get("duration_minutes") || "").trim() ? Number(form.get("duration_minutes")) : null,
      skills:String(form.get("skills") || "").split(",").map((s) => s.trim()).filter(Boolean),
      cv_text:String(form.get("cv_text") || "").trim() || null, job_description:String(form.get("job_description") || "").trim() || null,
      interview_format:String(form.get("interview_format") || "").trim() || null, interview_language:String(form.get("interview_language") || "vi"), metadata:{client:"FE-test",source:"human"},
    };
    const cvFile = form.get("cv_file");
    const jdFile = form.get("job_description_file");
    const useCv = form.get("profile_input_mode") === "cv";
    if (useCv && !payload.cv_text && !(cvFile instanceof File && cvFile.size)) {
      notice(t("Hãy tải tệp CV hoặc dán nội dung CV."), true);
      return;
    }
    if (!useCv && (!payload.candidate_name || !payload.target_role || !payload.target_company)) return;
    let creating = false;
    try {
      const maxUploadBytes = 12 * 1024 * 1024;
      if ([cvFile, jdFile].some((file) => file instanceof File && file.size > maxUploadBytes)) {
        notice(t("Mỗi tệp tải lên tối đa 12 MB."),true);
        return;
      }
      const uploads = {};
      if (cvFile instanceof File && cvFile.size) uploads.cv = await sha256(cvFile);
      if (jdFile instanceof File && jdFile.size) uploads.job_description = await sha256(jdFile);
      payload.metadata.upload_sha256 = uploads;
      const fingerprint = await sha256(JSON.stringify(payload));
      let prior = null;
      try { prior = JSON.parse(sessionStorage.getItem(key("pending-create", "new")) || "null"); } catch { /* ignore stale tab state */ }
      const request = prior?.fingerprint === fingerprint ? prior : {id:uuid(), fingerprint};
      sessionStorage.setItem(key("pending-create", "new"), JSON.stringify(request));
      state.creating = true;
      creating = true;
      renderProvider();
      $("create-submit").textContent = t("Đang tạo phiên…");
      const hasUpload = Boolean(uploads.cv || uploads.job_description);
      let body = JSON.stringify(payload);
      let path = "/api/v1/sessions";
      if (hasUpload) {
        const multipart = new FormData();
        multipart.append("payload", JSON.stringify(payload));
        if (uploads.cv) multipart.append("cv_file", cvFile, cvFile.name);
        if (uploads.job_description) multipart.append("job_description_file", jdFile, jdFile.name);
        body = multipart;
        path = "/api/v1/sessions/upload";
      }
      const result = await api(path, {method:"POST", headers:{"Idempotency-Key":request.id}, body});
      sessionStorage.removeItem(key("pending-create", "new"));
      $("create-dialog").close();
      formElement.reset();
      syncProfileInputMode();
      await refreshRuns();
      await selectRun(result.session_id);
      notice(t("Core đã chuẩn bị kế hoạch. Hãy xem lại vòng, thời lượng và giới hạn nguồn trước khi bắt đầu."));
    } catch (error) {
      notice(t("Không tạo được phiên: {error}. Gửi lại cùng hồ sơ sẽ dùng lại Idempotency-Key.",{error:error.message}),true);
    }
    finally {
      if (creating) {
        state.creating = false;
        $("create-submit").textContent = t("Tạo phiên và xem kế hoạch");
        renderProvider();
      }
    }
  });
  $("save-plan").addEventListener("click", async () => {
    if (!state.id || state.session?.status !== "READY") return;
    const duration = $("plan-duration").value.trim();
    const payload = {
      seniority:$("plan-seniority").value || null,
      interview_round:$("plan-round").value || null,
      duration_minutes:duration ? Number(duration) : null,
    };
    $("save-plan").disabled = true;
    try {
      await api(`/api/v1/sessions/${encodeURIComponent(state.id)}/plan`, {method:"PATCH",body:JSON.stringify(payload)});
      state.artifacts.clear();
      await hydrate();
      notice(t("Kế hoạch đã được cập nhật. Xem lại nguồn và giới hạn trước khi bắt đầu."));
    } catch (error) { notice(t("Không cập nhật được kế hoạch: {error}",{error:error.message}),true); }
    finally { renderSession(); }
  });
  $("edit-profile").addEventListener("click", () => {
    if (!state.candidateProfile || state.session?.status !== "READY") return;
    $("profile-json").value = JSON.stringify(state.candidateProfile,null,2);
    $("profile-dialog").showModal();
  });
  $("profile-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.id || state.session?.status !== "READY") return;
    let candidateProfile;
    try {
      candidateProfile = JSON.parse($("profile-json").value);
      if (!candidateProfile || Array.isArray(candidateProfile) || typeof candidateProfile !== "object") throw new Error("Expected a JSON object");
    } catch (error) { notice(t("JSON hồ sơ không hợp lệ: {error}",{error:error.message}),true); return; }
    $("save-profile").disabled = true;
    try {
      await api(`/api/v1/sessions/${encodeURIComponent(state.id)}/profile`, {method:"PATCH",body:JSON.stringify({candidate_profile:candidateProfile})});
      state.artifacts.clear();
      $("profile-dialog").close();
      await hydrate();
      notice(t("Đã lưu hồ sơ đã sửa. Core sẽ dùng thông tin này khi bắt đầu."));
    } catch (error) { notice(t("Không lưu được hồ sơ: {error}",{error:error.message}),true); }
    finally { $("save-profile").disabled = false; renderSession(); }
  });
  $("answer-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("answer").value.trim();
    if (!text || $("send-answer").disabled || state.pending) return;
    state.pending = {message_id:uuid(), candidate_answer:text, expected_turn:Number(state.session.current_turn || 0) + 1};
    state.pendingRetryCount = 0;
    localStorage.setItem(key("pending-answer"), JSON.stringify(state.pending));
    sendPending();
  });
  $("answer").addEventListener("input", () => {
    if (state.id) localStorage.setItem(key("draft"), $("answer").value);
    $("send-answer").disabled = !state.answerEnabled || !$("answer").value.trim();
  });
  $("answer").addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); $("answer-form").requestSubmit(); } });
  $("retry-answer").addEventListener("click", () => sendPending(false));
  $("start-session").addEventListener("click", startInterview);
  // Irreversible actions need a second click within 4 s instead of a blocking browser dialog.
  function confirmed(buttonId, label) {
    const button = $(buttonId), pending = state.confirming.get(buttonId);
    if (pending) clearTimeout(pending.timer);
    state.confirming.delete(buttonId);
    button.classList.remove("confirming");
    if (pending?.id === state.id) return true;
    const timer = setTimeout(() => { state.confirming.delete(buttonId); button.classList.remove("confirming"); renderSession(); }, 4000);
    state.confirming.set(buttonId, {id:state.id, timer});
    button.classList.add("confirming");
    button.textContent = t(label);
    return false;
  }
  $("cancel-session").addEventListener("click", async () => {
    if (!state.id || $("cancel-session").disabled) return;
    if (!confirmed("cancel-session", "Nhấn lại để hủy phiên")) return;
    $("cancel-session").disabled = true;
    try { await api(`/api/v1/sessions/${encodeURIComponent(state.id)}/cancel`, {method:"POST"}); await hydrate(); notice(t("Core đã hủy phiên. Dữ liệu đã lưu vẫn có thể inspect.")); }
    catch (error) { notice(error.message,true); renderSession(); }
  });
  $("end-session").addEventListener("click", async () => {
    if (!state.id || $("end-session").disabled) return;
    if (!confirmed("end-session", "Nhấn lại để kết thúc ngay")) return;
    const id = state.id, epoch = state.epoch;
    state.ending.set(id, "requesting");
    renderSession();
    try {
      await api(`/api/v1/sessions/${encodeURIComponent(id)}/end`, {method:"POST"});
      state.ending.set(id, "accepted");
      localStorage.removeItem(key("pending-answer", id));
      localStorage.removeItem(key("draft", id));
      if (epoch !== state.epoch || id !== state.id) return;
      if (state.pendingRetryTimer) clearTimeout(state.pendingRetryTimer);
      state.pendingRetryTimer = null;
      state.pendingRetryCount = 0;
      state.pending = null;
      state.busy = false;
      $("answer").value = "";
      notice(t("Core đang dừng các agent và tổng hợp bằng chứng hiện có. Báo cáo có thể chưa đủ dữ liệu."));
      await hydrate(epoch);
    } catch (error) {
      if (state.ending.get(id) === "requesting") state.ending.delete(id);
      if (epoch === state.epoch && id === state.id) notice(error.message, true);
    } finally { if (epoch === state.epoch && id === state.id) renderSession(); }
  });
  $("show-report").addEventListener("click", () => openReport().catch((error) => notice(error.message,true)));
  ["new-session","empty-create"].forEach((id) => $(id).addEventListener("click", () => { if (canRun()) $("create-dialog").showModal(); }));
  $("refresh-runs").addEventListener("click", () => refreshRuns());
  $("more-runs").addEventListener("click", async () => {
    $("more-runs").disabled = true;
    try { await refreshRuns(false,true); } finally { $("more-runs").disabled = false; }
  });
  $("search-runs").addEventListener("input", () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => {
      state.runSearchEpoch += 1;
      state.runQuery = $("search-runs").value.trim();
      state.runs = [];
      refreshRuns(true).catch((error) => notice(error.message,true));
    }, 250);
  });
  $("settings-button").addEventListener("click", () => {
    $("backend-url").value = state.base === location.origin ? "" : state.base;
    $("backend-token").value = state.token;
    $("settings-dialog").showModal();
  });
  $("settings-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    let base;
    try {
      const url = new URL($("backend-url").value.trim() || location.origin);
      if (!["http:","https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error();
      base = url.href.replace(/\/$/, "");
    } catch { notice(t("API URL phải là địa chỉ HTTP hoặc HTTPS hợp lệ, không chứa token."),true); return; }
    state.epoch += 1;
    state.id = null;
    state.session = null;
    state.profile = null;
    state.candidateProfile = null;
    state.runs = [];
    state.runSearchEpoch += 1;
    state.artifacts.clear();
    state.base = base;
    state.token = $("backend-token").value.trim();
    localStorage.setItem("core-studio-base-url", base);
    sessionStorage.setItem("core-studio-token", state.token);
    $("settings-dialog").close();
    await connect();
  });
  document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => $(button.dataset.close).close()));
  window.addEventListener("pagehide", disconnectStream);
  // A page restored from the back/forward cache has no stream; re-read backend state and reopen it.
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted || !state.connected || !state.id) return;
    const epoch = state.epoch;
    hydrate(epoch).then(() => { if (epoch === state.epoch && state.session && !TERMINAL.has(state.session.status) && canRunSelected()) openStream(epoch); }).catch((error) => notice(error.message,true));
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden || !state.connected || state.refreshing || !state.id) return;
    hydrate().catch((error) => notice(error.message,true));
  });
  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], {type}));
    const link = document.createElement("a");
    link.href = url; link.download = name;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $("download-transcript").addEventListener("click", () => {
    if (!state.id || !state.messages.length) return;
    const profile = state.profile || {};
    const lines = [`# ${t("Transcript phỏng vấn")} · ${profile.candidate_name || state.id}`, "", `- ${t("Mã phiên")}: ${state.id}`];
    if (profile.target_role) lines.push(`- ${t("Vị trí phỏng vấn")}: ${profile.target_role}`);
    lines.push(`- ${t("Trạng thái")}: ${state.session?.status || "—"}`, "");
    state.messages.forEach((m) => lines.push(`## ${t("Lượt")} ${m.turn} · ${m.role === "candidate" ? t("Bạn") : t("Core · Người phỏng vấn")}`, "", m.text, ""));
    download(`transcript_${state.id}.md`, lines.join("\n"), "text/markdown;charset=utf-8");
  });
  $("report-download").addEventListener("click", () => {
    if (!state.report) return;
    const markdown = typeof state.report.report_markdown === "string";
    download(`report_${state.id}.${markdown ? "md" : "json"}`, markdown ? state.report.report_markdown : JSON.stringify(state.report,null,2), markdown ? "text/markdown;charset=utf-8" : "application/json");
  });
  state.pollTimer = setInterval(async () => {
    if (!state.connected || state.refreshing || document.hidden) return;
    state.refreshing = true;
    try {
      // Finished sessions rarely change; avoid re-reading the full run detail every tick.
      const settled = TERMINAL.has(state.session?.status) && !state.ending.has(state.id);
      if (state.id && (!settled || Date.now() - state.lastHydrate > 30000)) await hydrate();
      await refreshRuns();
    }
    catch (error) { notice(t("Không đồng bộ được dữ liệu từ core: {error}",{error:error.message}),true); }
    finally { state.refreshing = false; }
  }, 3000);
  function applyLocale() {
    const previousNotice = translatedCopy.get($("notice").textContent);
    const previousWarning = translatedCopy.get(state.sourceWarning);
    document.documentElement.lang = i18n.locale;
    document.title = `Core Studio · ${t("Thử phỏng vấn")}`;
    $("interface-locale").value = i18n.locale;
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
    document.querySelectorAll("[placeholder], [title], [aria-label]").forEach((node) => {
      ["placeholder","title","aria-label"].forEach((attr) => {
        if (!node.hasAttribute(attr)) return;
        const sourceAttr = `data-i18n-${attr}`;
        if (!node.hasAttribute(sourceAttr)) node.setAttribute(sourceAttr,node.getAttribute(attr));
        node.setAttribute(attr,t(node.getAttribute(sourceAttr)));
      });
    });
    if (previousNotice) $("notice").textContent = t(previousNotice.source,previousNotice.values);
    if (previousWarning) state.sourceWarning = t(previousWarning.source,previousWarning.values);
    renderProvider();
    if (state.connected || state.runs.length) renderRuns();
    renderSession();
    $("create-submit").textContent = state.creating ? t("Đang tạo phiên…") : t("Tạo phiên và xem kế hoạch");
    if (state.report) renderReport();
  }
  $("interface-locale").addEventListener("change", (event) => {
    i18n.setLocale(event.target.value);
    applyLocale();
  });
  window.addEventListener("storage", (event) => {
    if (event.key !== "core-studio-locale") return;
    i18n.setLocale(event.newValue);
    applyLocale();
  });

  function markdownInline(text) {
    // Raw HTML is always escaped. Links are explicit safe protocols only; no image loading.
    const tokens = [];
    const token = (html) => { tokens.push(html); return `\u0000${tokens.length - 1}\u0000`; };
    let value = String(text).replace(/`([^`]+)`/g, (_, code) => token(`<code>${escape(code)}</code>`));
    value = value.replace(/\[([^\]]+)\]\(([^\s)]+)\)/g, (_, label, href) => {
      try {
        const url = new URL(href);
        if (!["https:","http:","mailto:"].includes(url.protocol)) return label;
        return token(`<a href="${escape(url.href)}" rel="noopener noreferrer" target="_blank">${escape(label)}</a>`);
      } catch { return label; }
    });
    value = escape(value).replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>").replace(/__([^_]+)__/g,"<strong>$1</strong>").replace(/\*([^*]+)\*/g,"<em>$1</em>");
    return value.replace(/\u0000(\d+)\u0000/g, (_, index) => tokens[Number(index)] || "");
  }
  function renderMarkdown(source) {
    const lines = String(source).replace(/\r\n?/g,"\n").split("\n");
    const out = [];
    let paragraph = [], list = null, code = null, language = "";
    const flushParagraph = () => { if (paragraph.length) out.push(`<p>${paragraph.map(markdownInline).join("<br>")}</p>`); paragraph = []; };
    const closeList = () => { if (list) out.push(`</${list}>`); list = null; };
    const cells = (line) => line.trim().replace(/^\||\|$/g,"").split("|").map((cell) => cell.trim());
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      const fence = line.match(/^\s*```(.*)$/);
      if (fence) {
        flushParagraph(); closeList();
        if (code === null) { code = []; language = fence[1].trim(); }
        else { out.push(`<div class="code-block"><div class="code-caption">${escape(language || "text")}</div><pre><code>${escape(code.join("\n"))}</code></pre></div>`); code = null; }
        continue;
      }
      if (code !== null) { code.push(line); continue; }
      if (!line.trim()) { flushParagraph(); closeList(); continue; }
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) { flushParagraph(); closeList(); out.push(`<h${heading[1].length}>${markdownInline(heading[2])}</h${heading[1].length}>`); continue; }
      if (/^\s*(?:---+|\*\*\*+)\s*$/.test(line)) { flushParagraph(); closeList(); out.push("<hr>"); continue; }
      if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
        flushParagraph(); closeList();
        out.push(`<div class="table-scroll"><table><thead><tr>${cells(line).map((cell) => `<th>${markdownInline(cell)}</th>`).join("")}</tr></thead><tbody>`);
        i += 1;
        while (i + 1 < lines.length && lines[i + 1].includes("|") && lines[i + 1].trim()) out.push(`<tr>${cells(lines[++i]).map((cell) => `<td>${markdownInline(cell)}</td>`).join("")}</tr>`);
        out.push("</tbody></table></div>"); continue;
      }
      const item = line.match(/^\s*(?:[-*+] |\d+[.)] )(.+)$/);
      if (item) { flushParagraph(); const nextList = /^\s*\d/.test(line) ? "ol" : "ul"; if (list !== nextList) { closeList(); list = nextList; out.push(`<${list}>`); } out.push(`<li>${markdownInline(item[1])}</li>`); continue; }
      const quote = line.match(/^>\s?(.*)$/);
      if (quote) { flushParagraph(); closeList(); out.push(`<blockquote>${markdownInline(quote[1])}</blockquote>`); continue; }
      closeList(); paragraph.push(line);
    }
    flushParagraph(); closeList();
    if (code !== null) out.push(`<div class="code-block"><div class="code-caption">${escape(language || "text")}</div><pre><code>${escape(code.join("\n"))}</code></pre></div>`);
    return out.join("");
  }
  function renderJson(value) {
    let count = 0, truncated = false;
    const node = (item, depth) => {
      count += 1;
      if (count > 5000 || depth > 20) { truncated = true; return `<span class="json-more">…</span>`; }
      if (item === null || typeof item !== "object") return `<span class="json-value json-${item === null ? "null" : typeof item}">${escape(JSON.stringify(item))}</span>`;
      const entries = Object.entries(item), array = Array.isArray(item);
      if (entries.length > 1000) truncated = true;
      return `<details class="json-node" ${depth < 2 ? "open" : ""}><summary><span class="json-bracket">${array ? "[ ]" : "{ }"}</span> <span class="json-count">${entries.length}</span></summary><div class="json-children">${entries.slice(0,1000).map(([key,value]) => `<div class="json-entry"><span class="json-key">${escape(array ? key : JSON.stringify(key))}</span>${node(value,depth+1)}</div>`).join("")}${entries.length > 1000 ? '<span class="json-more">…</span>' : ""}</div></details>`;
    };
    const tree = node(value,0);
    return (truncated ? `<p class="preview-limit">${escape(t("Cây JSON được giới hạn để giữ giao diện mượt. Mở Nội dung gốc để đọc dữ liệu đầy đủ."))}</p>` : "") + tree;
  }
  // Chỉ PASS (sau khi tính review của người duyệt) mới xanh; FAIL và NEEDS_REVIEW đều là chưa đạt.
  function renderVerdict(report) {
    const assessment = report.final_assessment || {};
    const decision = report.decision || {};
    const verdict = report.verdict || decision.verdict || assessment.verdict;
    if (!verdict) return "";
    const passed = verdict === "PASS";
    const score = decision.weighted_score ?? assessment.metrics?.weighted_score;
    const threshold = decision.pass_threshold ?? assessment.metrics?.pass_threshold;
    const summary = assessment.narrative?.summary || (decision.verdict_reasons || []).join(" ");
    const note = verdict === "NEEDS_REVIEW" ? t("Kết quả nằm ở vùng ranh giới hoặc thiếu bằng chứng; cần người duyệt xác nhận.") : "";
    const scoreText = typeof score === "number" ? t("Điểm {score}/4 · ngưỡng đạt {threshold}", {score, threshold: threshold ?? "—"}) : "";
    return `<section class="verdict-banner ${passed ? "pass" : "fail"}" role="status"><div class="verdict-mark" aria-hidden="true">${passed ? "✓" : "✕"}</div><div><strong>${escape(passed ? t("ĐẠT PVAN") : t("KHÔNG ĐẠT PVAN"))}</strong>${scoreText ? `<span class="verdict-score">${escape(scoreText)}</span>` : ""}${summary ? `<p>${escape(summary)}</p>` : ""}${note ? `<p class="verdict-note">${escape(note)}</p>` : ""}</div></section>`;
  }
  function renderReport() {
    const report = state.report;
    if (!report) return;
    const markdown = typeof report.report_markdown === "string" ? report.report_markdown : null;
    $("report-content").className = `document-view ${markdown === null ? "json-view" : "markdown-view"}`;
    $("report-content").innerHTML = markdown === null ? renderJson(report) : `${renderMarkdown(markdown)}<details class="report-envelope"><summary>${escape(t("Toàn bộ phản hồi JSON từ core"))}</summary><div class="json-view">${renderJson(report)}</div></details>`;
    if (report.final_assessment?.early_ended) $("report-content").insertAdjacentHTML("afterbegin", `<p class="notice">${escape(t("Phiên kết thúc sớm. Báo cáo chỉ dựa trên bằng chứng đã lưu; một số câu trả lời có thể chưa được đánh giá."))}</p>`);
    $("report-content").insertAdjacentHTML("afterbegin", renderVerdict(report));
    $("report-raw").textContent = JSON.stringify(report,null,2);
    $("report-copy").textContent = t("Sao chép");
  }
  function setReportView(source) {
    $("report-content").hidden = source;
    $("report-raw").hidden = !source;
    ["report-rendered","report-source"].forEach((id) => {
      const active = (id === "report-source") === source;
      $(id).classList.toggle("active",active);
      $(id).setAttribute("aria-pressed",String(active));
    });
  }
  $("report-rendered").addEventListener("click", () => setReportView(false));
  $("report-source").addEventListener("click", () => setReportView(true));
  $("report-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("report-raw").textContent); $("report-copy").textContent = t("Đã sao chép"); }
    catch { notice(t("Không thể sao chép; chọn nội dung gốc để sao chép thủ công."),true); }
  });
  applyLocale();
  connect();
})();
