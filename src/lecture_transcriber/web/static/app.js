// Minimal, framework-free client. Uses fetch + DOM APIs only.
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  function text(el, value) {
    if (el) el.textContent = value == null ? "" : String(value);
  }

  function fmtBytes(n) {
    if (n == null) return "";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i ? 1 : 0) + " " + u[i];
  }

  async function loadSystem() {
    try {
      const r = await fetch("/api/system");
      if (!r.ok) return;
      const s = await r.json();
      text($("info-data-dir"), s.data_dir);
      text($("info-offline"), s.offline ? "yes" : "no");
      text($("info-max-upload"), fmtBytes(s.max_upload_bytes));
      const h = s.hardware || {};
      text($("info-cpu"), h.cpu_count);
      text($("info-ram"), fmtBytes(h.ram_bytes));
      text($("info-cuda"), h.cuda_available ? (h.cuda_name || "yes") : "no");
      text($("info-engine"), s.asr_engine);
      text($("info-version"), s.asr_version);
      const sel = $("model");
      if (sel) {
        for (const m of s.available_models || []) {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          sel.appendChild(opt);
        }
      }
    } catch (e) { /* ignore */ }
  }

  async function loadJobs() {
    const body = $("jobs-body");
    const empty = $("jobs-empty");
    if (!body) return;
    try {
      const r = await fetch("/api/jobs?limit=20");
      if (!r.ok) return;
      const items = await r.json();
      body.textContent = "";
      if (!items.length) {
        if (empty) empty.hidden = false;
        return;
      }
      if (empty) empty.hidden = true;
      for (const j of items) {
        const tr = document.createElement("tr");
        const link = document.createElement("a");
        link.href = "/jobs/" + j.id;
        link.textContent = j.id;
        const tdId = document.createElement("td");
        tdId.appendChild(link);
        const tdMedia = document.createElement("td");
        tdMedia.textContent = j.media_name || "";
        const tdStatus = document.createElement("td");
        tdStatus.textContent = j.status;
        const tdProg = document.createElement("td");
        tdProg.textContent = (j.progress || 0) + "%";
        const tdCreated = document.createElement("td");
        tdCreated.textContent = j.created_at ? new Date(j.created_at).toLocaleString() : "";
        tr.append(tdId, tdMedia, tdStatus, tdProg, tdCreated);
        body.appendChild(tr);
      }
    } catch (e) { /* ignore */ }
  }

  async function upload(form) {
    const status = $("upload-status");
    const file = $("file").files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    text(status, "uploading…");
    const r = await fetch("/api/media", { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      text(status, (body.error && body.error.message) || "upload failed");
      return;
    }
    const mediaId = body.media.id;
    text(status, "creating job…");
    const jr = await fetch("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        media_id: mediaId,
        language: $("language").value || null,
        model_override: $("model").value || null,
      }),
    });
    const jbody = await jr.json().catch(() => ({}));
    if (!jr.ok) {
      text(status, (jbody.error && jbody.error.message) || "job create failed");
      return;
    }
    window.location.href = "/jobs/" + jbody.id;
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadSystem();
    loadJobs();
    setInterval(loadJobs, 5000);
    const form = $("upload-form");
    if (form) form.addEventListener("submit", (e) => { e.preventDefault(); upload(form); });
  });
})();
