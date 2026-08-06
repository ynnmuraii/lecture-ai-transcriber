// Job detail page: polls /api/jobs/{id} every 2s until terminal.
(function () {
  "use strict";

  const editor = { revision: 0, segments: [], drafts: new Map() };

  function $(id) { return document.getElementById(id); }
  function text(el, value) { if (el) el.textContent = value == null ? "" : String(value); }
  function fmtTime(s) {
    if (s == null) return "";
    const m = Math.floor(s / 60);
    const r = (s - m * 60).toFixed(2);
    return m + ":" + r.padStart(5, "0");
  }
  function isTerminal(status) {
    return status === "completed" || status === "completed_with_warnings" ||
      status === "failed" || status === "cancelled";
  }
  function jobId() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    return parts[parts.length - 1];
  }
  function editorHasChanges() {
    return editor.segments.some((segment) => {
      return editor.drafts.has(segment.id) &&
        editor.drafts.get(segment.id) !== (segment.text || "");
    });
  }
  function showEditorError(message) {
    const error = $("editor-error");
    if (!error) return;
    error.hidden = !message;
    error.textContent = message || "";
  }
  function renderSegments(segments) {
    const body = $("segments-body");
    const empty = $("segments-empty");
    if (!body) return;
    body.textContent = "";
    if (!segments.length) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    for (const segment of segments) {
      const tr = document.createElement("tr");
      const index = document.createElement("td");
      index.textContent = segment.index;
      const time = document.createElement("td");
      time.textContent = fmtTime(segment.start) + "–" + fmtTime(segment.end);
      const speaker = document.createElement("td");
      speaker.textContent = segment.speaker_id || "—";
      const textCell = document.createElement("td");
      const textarea = document.createElement("textarea");
      textarea.rows = 2;
      textarea.className = "segment-editor";
      textarea.value = editor.drafts.has(segment.id)
        ? editor.drafts.get(segment.id)
        : (segment.text || "");
      textarea.setAttribute("aria-label", "Text for segment " + segment.index);
      textarea.addEventListener("input", () => {
        editor.drafts.set(segment.id, textarea.value);
        $("save-editor").disabled = !editorHasChanges();
      });
      textCell.appendChild(textarea);
      if (segment.polished_text) {
        const preview = document.createElement("small");
        preview.className = "polished-preview";
        preview.textContent = "Polished preview: " + segment.polished_text;
        textCell.appendChild(preview);
      }
      const warning = document.createElement("td");
      if (segment.needs_review) {
        warning.textContent = "review";
        warning.className = "warn";
      }
      tr.append(index, time, speaker, textCell, warning);
      body.appendChild(tr);
    }
    $("save-editor").disabled = !editorHasChanges();
  }
  async function loadEditor(id) {
    const response = await fetch("/api/jobs/" + id + "/editor");
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) return false;
    const unsaved = editorHasChanges();
    const changedRevision = payload.revision !== editor.revision;
    editor.revision = payload.revision;
    editor.segments = payload.segments || [];
    if (!unsaved) editor.drafts = new Map();
    if (changedRevision || !unsaved) renderSegments(editor.segments);
    text($("editor-revision"), "Revision " + editor.revision);
    showEditorError("");
    return true;
  }
  async function saveEditor(id) {
    if (!editorHasChanges()) return;
    const edits = editor.segments
      .filter((segment) => editor.drafts.has(segment.id) &&
        editor.drafts.get(segment.id) !== (segment.text || ""))
      .map((segment) => ({
        segment_id: segment.id,
        text: editor.drafts.get(segment.id)
      }));
    $("save-editor").disabled = true;
    showEditorError("");
    const response = await fetch("/api/jobs/" + id + "/editor", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_revision: editor.revision, edits: edits })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      showEditorError((payload.error && payload.error.message) ||
        "Save failed; reload to resolve the revision conflict");
      $("save-editor").disabled = false;
      return;
    }
    editor.revision = payload.revision;
    editor.segments = payload.segments || [];
    editor.drafts = new Map();
    renderSegments(editor.segments);
    text($("editor-revision"), "Saved revision " + editor.revision);
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
    const artifactNames = {
      speaker: "speaker.json",
      speaker_txt: "speaker.txt",
      polished: "polished.json",
      editor: "editor.json",
    };
    for (const a of arts) {
      const li = document.createElement("li");
      const link = document.createElement("a");
      const name = artifactNames[a.format] || "transcript." + a.format;
      link.href = "/api/artifacts/" + a.id;
      link.textContent = name;
      link.download = name;
      li.appendChild(link);
      list.appendChild(li);
    }
  }
  async function load() {
    const id = jobId();
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

    const editorLoaded = await loadEditor(id).catch(() => false);
    const arts = d.artifacts || [];
    if (!editorLoaded) {
      const jsonArt = arts.find((a) => a.format === "json");
      if (jsonArt) {
        try {
          const ar = await fetch("/api/artifacts/" + jsonArt.id);
          if (ar.ok) {
            const tj = await ar.json();
            editor.segments = tj.segments || [];
            renderSegments(editor.segments);
          }
        } catch (e) { /* raw preview is best effort while editor is unavailable */ }
      }
    }
    renderArtifacts(arts);
    return isTerminal(d.status);
  }
  async function cancel() {
    await fetch("/api/jobs/" + jobId() + "/cancel", { method: "POST" });
    load();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const cancelButton = $("cancel-btn");
    if (cancelButton) cancelButton.addEventListener("click", cancel);
    const saveButton = $("save-editor");
    if (saveButton) saveButton.addEventListener("click", () => {
      saveEditor(jobId()).catch(() => showEditorError("Save failed"));
    });
    let stopped = false;
    (async () => {
      while (!stopped) {
        const done = await load().catch(() => true);
        if (done) { stopped = true; break; }
        await new Promise((r) => setTimeout(r, 2000));
      }
    })();
  });
})();
