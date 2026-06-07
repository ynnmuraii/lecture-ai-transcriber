// Job detail page: polls /api/jobs/{id} every 2s until terminal.
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }
  function text(el, value) { if (el) el.textContent = value == null ? "" : String(value); }
  function fmtTime(s) {
    if (s == null) return "";
    const m = Math.floor(s / 60);
    const r = (s - m * 60).toFixed(2);
    return m + ":" + r.padStart(5, "0");
  }

  function isTerminal(status) {
    return status === "completed" || status === "failed" || status === "cancelled";
  }

  async function load() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    const id = parts[parts.length - 1];
    const r = await fetch("/api/jobs/" + id);
    if (!r.ok) {
      text($("job-status"), "not found");
      return true;
    }
    const d = await r.json();
    text($("job-media"), d.media_name);
    text($("job-status"), d.status);
    text($("job-progress"), d.progress + "%");
    text($("job-stage"), d.stage_message);
    text($("job-profile"), d.profile_name || "(auto)");
    text($("job-cancel"), d.cancel_requested ? "yes" : "no");

    const err = $("job-error");
    if (d.error_message) {
      err.hidden = false;
      err.textContent = (d.error_code || "ERROR") + ": " + d.error_message;
    } else {
      err.hidden = true;
    }

    const segs = d.events || [];
    // The events list is the journal; the canonical JSON gives segments. We
    // pull the JSON artifact (if it exists) and render its segments.
    const arts = d.artifacts || [];
    const jsonArt = arts.find((a) => a.format === "json");
    if (jsonArt) {
      try {
        const ar = await fetch("/api/artifacts/" + jsonArt.id);
        if (ar.ok) {
          const tj = await ar.json();
          renderSegments(tj.segments || []);
        }
      } catch (e) { /* ignore */ }
    }
    renderArtifacts(arts);

    return isTerminal(d.status);
  }

  function renderSegments(segs) {
    const body = $("segments-body");
    const empty = $("segments-empty");
    if (!body) return;
    body.textContent = "";
    if (!segs.length) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    for (const s of segs) {
      const tr = document.createElement("tr");
      const tdI = document.createElement("td");
      tdI.textContent = s.index;
      const tdS = document.createElement("td");
      tdS.textContent = fmtTime(s.start);
      const tdE = document.createElement("td");
      tdE.textContent = fmtTime(s.end);
      const tdT = document.createElement("td");
      tdT.textContent = s.text;
      const tdW = document.createElement("td");
      if (s.needs_review) {
        tdW.textContent = "⚠";
        tdW.className = "warn";
        tdW.title = (s.warnings || []).join("; ");
      }
      tr.append(tdI, tdS, tdE, tdT, tdW);
      body.appendChild(tr);
    }
  }

  function renderArtifacts(arts) {
    const list = $("artifacts");
    const empty = $("artifacts-empty");
    if (!list) return;
    list.textContent = "";
    if (!arts.length) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    for (const a of arts) {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = "/api/artifacts/" + a.id;
      link.textContent = "transcript." + a.format;
      link.download = "transcript." + a.format;
      li.appendChild(link);
      list.appendChild(li);
    }
  }

  async function cancel() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    const id = parts[parts.length - 1];
    await fetch("/api/jobs/" + id + "/cancel", { method: "POST" });
    load();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const c = $("cancel-btn");
    if (c) c.addEventListener("click", cancel);
    let stopped = false;
    (async () => {
      while (!stopped) {
        const done = await load().catch(() => true);
        if (done) break;
        await new Promise((r) => setTimeout(r, 2000));
      }
    })();
  });
})();
