(() => {
  "use strict";
  const messages = {
    "Ứng dụng": "Applications", "Inspect": ["Giám sát", "Inspect"], "Test core ↗": ["Chạy thử core ↗", "Test core ↗"],
    "Ngôn ngữ giao diện": "Interface language", "Cài đặt backend nâng cao": "Advanced backend settings",
    "Danh sách lượt chạy": "Run list", "Lượt chạy": "Runs", "Làm mới danh sách": "Refresh runs", "Tìm lượt chạy": "Search runs",
    "Tìm tên, vai trò hoặc session ID": "Search name, role or session ID", "Trạng thái": "Status", "Tất cả": "All",
    "Đang chạy": "Running", "Hoàn tất": "Completed", "Có lỗi": "With errors", "Đang tải dữ liệu từ core…": "Loading data from core…",
    "Tải thêm lượt chạy": "Load more runs", "Lưu trữ: chưa xác định": "Storage: unknown", "Provider: chưa xác định": ["Nhà cung cấp: chưa xác định", "Provider: unknown"],
    "Trace của từng lượt chạy": ["Theo dõi chi tiết từng lượt chạy", "Inspect every run"],
    "Chọn một session để xem cây agent, lời gọi và thời gian sống đã được core ghi lại.": ["Chọn phiên để xem cây agent, lời gọi và vòng đời đã được core ghi lại.", "Choose a session to explore recorded agents, calls and lifetimes."],
    "Mở FE-test để chạy core ↗": "Open FE-test to run core ↗",
    "Dữ liệu hiển thị được đọc từ backend. Lượt chạy chưa ghi trace sẽ được đánh dấu rõ.": "Displayed data comes from backend records. Runs without tracing are explicitly marked.",
    "Sao chép session ID": "Copy session ID", "Dữ liệu run": ["Dữ liệu lượt chạy", "Run data"],
    "Flow map": ["Sơ đồ luồng", "Flow map"], "Waterfall": ["Dòng thời gian", "Waterfall"], "Events": ["Sự kiện", "Events"], "Artifacts": ["Tệp kết quả", "Artifacts"],
    "Đang đồng bộ": "Synchronizing", "Dữ liệu trace": ["Dữ liệu theo dõi", "Trace data"],
    "Quan hệ parent → child được ghi bởi runtime": ["Quan hệ cha → con do runtime ghi lại", "Parent → child relationships recorded by runtime"],
    "Timing thực tế · agent và lời gọi theo hierarchy": ["Thời gian thực tế · agent và lời gọi theo cây", "Recorded timing · agents and calls in hierarchy"],
    "Mở tất cả": "Expand all", "Thu nhỏ flow": ["Thu nhỏ sơ đồ", "Zoom out"], "Phóng to flow": ["Phóng to sơ đồ", "Zoom in"], "Vừa khung": "Fit view",
    "Thu phóng timeline": "Timeline zoom", "Kéo nền để di chuyển ngang": "Drag background to pan horizontally",
    "Agent": "Agent", "LLM": "LLM", "Tool": ["Công cụ", "Tool"], "Retrieval": ["Truy xuất", "Retrieval"],
    "Task / operation": ["Tác vụ / thao tác", "Task / operation"], "Đường nối = parent_span_id đã ghi": "Edges = recorded parent_span_id",
    "Chi tiết span": ["Chi tiết bước", "Span details"], "Chọn một bước": "Select a step",
    "Xem input, output, lỗi và lifetime của agent hoặc lời gọi.": ["Xem đầu vào, đầu ra, lỗi và vòng đời của agent hoặc lời gọi.", "View inputs, outputs, errors and agent or call lifetime."],
    "Backend nâng cao": "Advanced backend", "Đóng": "Close",
    "UI tự dùng core cùng địa chỉ. Chỉ thay đổi khi inspect một backend khác.": ["Giao diện tự dùng core cùng địa chỉ. Chỉ thay đổi khi giám sát backend khác.", "The UI uses core at the same origin. Change only to inspect another backend."],
    "API base URL": ["Địa chỉ gốc API", "API base URL"], "Service token": ["Token dịch vụ", "Service token"],
    "Để trống để dùng token phát triển nếu được bật": "Leave blank to use a development token if enabled",
    "Cấu hình dùng chung với FE-test. Token được lưu trong session của tab.": ["Cấu hình dùng chung với FE-test. Token được lưu trong phiên của tab.", "Configuration is shared with FE-test. The token is stored in the tab session."],
    "Dùng core hiện tại": "Use current core", "Lưu cấu hình": "Save configuration",
    "Khởi tạo": "Initialized", "Chuẩn bị": "Preparing", "Sẵn sàng": "Ready", "Chuyển giai đoạn": "Stage transition", "Kết thúc hội thoại": "Dialogue completed",
    "Đánh giá": "Assessing", "Lỗi": "Failed", "Lỗi đánh giá": "Assessment failed", "Đã hủy": "Cancelled", "Hết hạn": "Expired", "Đã lưu": "Archived",
    "Gián đoạn": "Interrupted", "Timeout": ["Quá thời hạn", "Timed out"], "Suy giảm": "Degraded", "Chờ chạy": "Queued",
    "Chưa ghi trạng thái": "Status not recorded", "Chưa ghi": "Not recorded", "Chưa ghi thời gian": "Timing not recorded",
    "Không lấy được token local mới.": "Could not obtain a new local token.", "Cấu hình kết nối đã thay đổi; hãy kết nối lại.": "Connection settings changed; reconnect.",
    "Trace chưa ghi": ["Chưa ghi trace", "Trace not recorded"], "Không có lượt chạy khớp bộ lọc.": "No runs match the filter.",
    "Chưa có lượt chạy được lưu. Mở FE-test để chạy core.": "No saved runs yet. Open FE-test to run core.", "Không đọc được core": "Cannot read core",
    "Đang đọc dữ liệu đã lưu từ core…": "Reading saved data from core…", "Đang đọc trace": ["Đang đọc dữ liệu theo dõi", "Reading trace"],
    "Dữ liệu được tải từ backend.": "Data is loaded from the backend.", "Đã đồng bộ · 1,8 giây/lần": "Synchronized · every 1.8 seconds", "Lỗi tải dữ liệu": "Data loading failed",
    "Vai trò chưa ghi": "Role not recorded", "Spans đã ghi": ["Bước đã ghi", "Recorded spans"], "Agent instances": ["Instance agent", "Agent instances"], "Run duration": ["Thời gian phiên", "Run duration"],
    "Archive simulation từ phiên bản cũ · dữ liệu cũ được giữ nguyên, không phải lượt chạy provider thật hiện tại.": ["Phiên mô phỏng từ bản cũ · giữ nguyên dữ liệu lịch sử, không phải lượt chạy nhà cung cấp thật hiện tại.", "Legacy simulation archive · historical data is preserved; this is not a current real-provider run."],
    "Các bước đã được runtime ghi lại. Trace bắt đầu từ lúc instrumentation được bật; bước cũ chưa ghi sẽ không được suy diễn.": ["Các bước đã được runtime ghi lại. Trace bắt đầu khi bật ghi nhận; không suy diễn bước cũ bị thiếu.", "Steps recorded by runtime. Tracing starts when instrumentation is enabled; missing earlier steps are not inferred."],
    "Trace lịch sử chưa đầy đủ. Chỉ hiển thị các bước và thời gian đã ghi; không suy diễn bước thiếu.": "Historical tracing is partial. Only recorded steps and timing are shown; missing steps are not inferred.",
    "Run này chưa có trace lifecycle. Events và artifacts vẫn có thể inspect; flow không được dựng giả.": ["Phiên chưa có trace vòng đời. Vẫn xem được sự kiện và tệp kết quả; không dựng sơ đồ giả.", "This run has no lifecycle trace. Events and artifacts remain available; no flow is fabricated."],
    "chưa xác định": "unknown", "Lưu trữ": "Storage", "Mở": "Expand", "Thu gọn": "Collapse", "bước con": "child spans",
    "Flow map từ trace runtime": ["Sơ đồ luồng từ trace runtime", "Flow map from runtime trace"], "Span / agent hierarchy": ["Cây bước / agent", "Span / agent hierarchy"],
    "Không có timing đã ghi": ["Chưa ghi thời gian", "No recorded timing"], "Thu gọn hoặc mở": "Collapse or expand", "Chưa ghi start/end": ["Chưa ghi bắt đầu / kết thúc", "Start/end not recorded"],
    "Chưa có trace lifecycle cho run này": ["Phiên chưa có trace vòng đời", "No lifecycle trace for this run"],
    "Core chưa ghi các span có quan hệ parent/child ở lượt chạy này. Chọn Events hoặc Artifacts để xem dữ liệu còn lưu.": ["Core chưa ghi các bước cha/con của phiên này. Chọn Sự kiện hoặc Tệp kết quả để xem dữ liệu đã lưu.", "Core has no recorded parent/child spans for this run. Choose Events or Artifacts to view saved data."],
    "Flow và lifetime sẽ xuất hiện khi có trace được ghi từ một lượt chạy mới.": ["Luồng và vòng đời sẽ xuất hiện khi lượt chạy mới được ghi trace.", "Flow and lifetime appear when a new run records traces."],
    "Lọc nội dung events đã tải": ["Lọc nội dung sự kiện đã tải", "Filter loaded events"], "Tải thêm events từ backend": ["Tải thêm sự kiện từ backend", "Load more events from backend"],
    "Chưa có artifact được lưu cho run này.": ["Chưa lưu tệp kết quả cho phiên này.", "No saved artifacts for this run."], "đã tải": "loaded", "Không có event khớp bộ lọc.": ["Không có sự kiện khớp bộ lọc.", "No events match the filter."],
    "Không có dữ liệu được ghi cho mục này.": "No recorded data for this section.", "Call lifecycle": ["Vòng đời lời gọi", "Call lifecycle"], "Instance ID": ["ID instance", "Instance ID"],
    "Lifetime scope": ["Phạm vi vòng đời", "Lifetime scope"], "Parent": ["Bước cha", "Parent"], "Scheduled": ["Đã lên lịch", "Scheduled"], "Started": ["Bắt đầu", "Started"], "Ended": ["Kết thúc", "Ended"],
    "Duration": ["Thời lượng", "Duration"], "Queue time": ["Thời gian chờ", "Queue time"], "Sequence": ["Thứ tự", "Sequence"], "Root · không có parent": ["Gốc · không có bước cha", "Root · no parent"],
    "Parent chưa tải": ["Bước cha chưa tải", "Parent not loaded"], "Application · dùng chung toàn tiến trình": ["Ứng dụng · dùng chung toàn tiến trình", "Application · shared across process"],
    "Session · tồn tại trong phiên": ["Phiên · tồn tại trong phiên", "Session · session lifetime"], "Delegated call · được tạo cho lời gọi": ["Ủy quyền · tạo cho lời gọi", "Delegated call · created for call"],
    "Agent instance · observation & release": ["Instance agent · ghi nhận và giải phóng", "Agent instance · observation & release"],
    "Registry status": ["Trạng thái registry", "Registry status"], "Observed": ["Ghi nhận", "Observed"], "Released": ["Giải phóng", "Released"], "Release reason": ["Lý do giải phóng", "Release reason"],
    "Scope": ["Phạm vi", "Scope"], "Process ID": ["ID tiến trình", "Process ID"], "Không biết thời điểm · process restart": ["Không rõ thời điểm · tiến trình khởi động lại", "Time unknown · process restarted"],
    "Observed là thời điểm runtime ghi nhận instance. Released là lúc registry trả instance; không phải thời điểm constructor hay destructor.": ["Ghi nhận là lúc runtime thấy instance. Giải phóng là lúc registry trả instance; không phải thời điểm tạo hay hủy đối tượng.", "Observed is when runtime records the instance. Released is registry release, not constructor or destructor timing."],
    "Error": ["Lỗi", "Error"], "Input": ["Đầu vào", "Input"], "Output": ["Đầu ra", "Output"], "Attributes": ["Thuộc tính", "Attributes"],
    "Timing lấy từ timestamp của backend. Ký hiệu ↻ là thời gian đang chạy tính từ started_at đã ghi. Task chỉ có scheduled_at đang chờ chạy; UI không tạo started_at.": ["Thời gian lấy từ backend. ↻ tính từ started_at đã ghi. Tác vụ chỉ có scheduled_at còn đang chờ; giao diện không tự tạo started_at.", "Timing uses backend timestamps. ↻ measures elapsed time from recorded started_at. Tasks with only scheduled_at are queued; the UI never invents started_at."],
    "EVENT": ["SỰ KIỆN", "EVENT"], "Timestamp": ["Thời điểm", "Timestamp"], "Event payload": ["Nội dung sự kiện", "Event payload"],
    "PERSISTED ARTIFACT": ["TỆP ĐÃ LƯU", "PERSISTED ARTIFACT"], "Size": ["Dung lượng", "Size"], "Modified": ["Cập nhật", "Modified"], "Đang đọc file từ backend…": ["Đang đọc tệp từ backend…", "Reading file from backend…"],
    "Đã đồng bộ": "Synchronized", "Không đồng bộ được": "Sync failed", "Đang kết nối core…": "Connecting to core…", "Đang kết nối…": "Connecting…",
    "Provider chưa cấu hình": ["Chưa cấu hình nhà cung cấp", "Provider not configured"], "không phải provider thật": ["không phải nhà cung cấp thật", "not a real provider"],
    "Core cùng địa chỉ": "Core at same origin", "Không kết nối được": "Connection failed", "Mở cài đặt nâng cao nếu backend yêu cầu service token.": ["Mở cài đặt nâng cao nếu backend yêu cầu token dịch vụ.", "Open advanced settings if the backend requires a service token."],
    "Không đọc được run": ["Không đọc được phiên", "Cannot read run"], "Sao chép": "Copy", "Không tải được events": ["Không tải được sự kiện", "Cannot load events"],
    "Không đọc được artifact": ["Không đọc được tệp", "Cannot read artifact"], "Đã sao chép session ID": "Session ID copied", "Trình duyệt không cho phép sao chép": "Browser blocked clipboard access", "Backend phải dùng http hoặc https": "Backend must use http or https",
    "Xem đẹp": "Read", "Nguồn": "Source", "Mở cây": "Expand tree", "Thu cây": "Collapse tree", "Nội dung đã lưu": "Saved content",
    "Hoàn thành": "Completed", "Đang kết thúc": "Ending", "Không xuất được run": "Cannot export run", "Tải toàn bộ run (JSON)": "Download full run (JSON)", "Đang chạy / chờ": "Running / queued", "Lỗi / gián đoạn": "Failed / interrupted",
  };
  let locale = localStorage.getItem("core-studio-locale") === "en" ? "en" : "vi";
  const t = (key) => {
    const item = messages[key];
    return Array.isArray(item) ? item[locale === "en" ? 1 : 0] : locale === "en" ? item || key : key;
  };
  function applyStatic() {
    document.documentElement.lang = locale; document.title = locale === "vi" ? "Core Studio · Giám sát" : "Core Studio · Inspect";
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
    document.querySelectorAll("[data-i18n-label]").forEach((node) => { node.setAttribute("aria-label", t(node.dataset.i18nLabel)); });
    document.querySelectorAll("[data-i18n-title]").forEach((node) => { node.title = t(node.dataset.i18nTitle); });
    const select = document.querySelector("#ui-language"); if (select) select.value = locale;
  }
  // Capture only initial static markup, before backend values are ever inserted.
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT), nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const key = node.textContent.trim(); if (!messages[key]) return;
    const parent = node.parentElement;
    if (parent.childNodes.length === 1) parent.dataset.i18n = key;
    else { const span = document.createElement("span"); span.dataset.i18n = key; span.textContent = key; node.replaceWith(span); }
  });
  document.querySelectorAll("[aria-label], [placeholder], [title]").forEach((node) => {
    for (const [attr, target] of [["aria-label", "i18nLabel"], ["placeholder", "i18nPlaceholder"], ["title", "i18nTitle"]]) { const key = node.getAttribute(attr); if (messages[key]) node.dataset[target] = key; }
  });
  window.CoreInspectI18n = {
    t, get locale() { return locale; }, get dateLocale() { return locale === "vi" ? "vi-VN" : "en-GB"; },
    set(value) { locale = value === "en" ? "en" : "vi"; localStorage.setItem("core-studio-locale", locale); applyStatic(); }, applyStatic,
  };
  applyStatic();
})();
