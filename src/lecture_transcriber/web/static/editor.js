// Derived transcript editor with optimistic revision saves.
(function () {
  "use strict";

  const state = { revision: 0, segments: [], drafts: new Map() };
  function $(id) { return document.getElementById(id); }
  function fmtTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = (seconds - minutes * 60).toFixed(2);
    return minutes + ":" + remainder.padStart(5, "0");
  }
  function jobId() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    return parts[parts.length - 2];
  }
  function setError(message) {
    const error = $("editor-error");
    if (!error) return;
    error.hidden = !message;
    error.textContent = message || "";
  }
  function setStatus(message) { $("editor-status").textContent = message; }
  function hasChanges() {
    return state.segments.some((segment) => {
      return state.drafts.has(segment.id) && state.drafts.get(segment.id) !== segment.text;
    });
  }
  function render() {
    const body = $("editor-segments-body");
    const empty = $("editor-empty");
    body.textContent = "";
    empty.hidden = state.segments.length !== 0;
    for (const segment of state.segments) {
      const row = document.createElement("tr");
      const index = document.createElement("td");
      index.textContent = segment.index;
      const time = document.createElement("td");
      time.textContent = fmtTime(segment.start) + "–" + fmtTime(segment.end);
      const speaker = document.createElement("td");
      speaker.textContent = segment.speaker_id || "—";
      const textCell = document.createElement("td");
      const textarea = document.createElement("textarea");
      textarea.rows = 2;
      textarea.value = state.drafts.has(segment.id) ? state.drafts.get(segment.id) : segment.text;
      textarea.dataset.segmentId = segment.id;
      textarea.addEventListener("input", () => {
        state.drafts.set(segment.id, textarea.value);
        $("editor-save").disabled = !hasChanges();
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
      row.append(index, time, speaker, textCell, warning);
      body.appendChild(row);
    }
    $("editor-save").disabled = !hasChanges();
  }
  async function load() {
    setError("");
    setStatus("Loading…");
    const response = await fetch("/api/jobs/" + jobId() + "/editor");
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setStatus("Editor unavailable");
      setError((payload.error && payload.error.message) || "Could not load editor");
      $("editor-save").disabled = true;
      return;
    }
    state.revision = payload.revision;
    state.segments = payload.segments || [];
    state.drafts = new Map();
    render();
    setStatus("Revision " + state.revision + " · " + state.segments.length + " segments");
  }
  async function save() {
    if (!hasChanges()) return;
    const edits = state.segments
      .filter((segment) => state.drafts.has(segment.id) && state.drafts.get(segment.id) !== segment.text)
      .map((segment) => ({ segment_id: segment.id, text: state.drafts.get(segment.id) }));
    $("editor-save").disabled = true;
    setError("");
    const response = await fetch("/api/jobs/" + jobId() + "/editor", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_revision: state.revision, edits: edits })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setError((payload.error && payload.error.message) || "Save failed; reload before retrying");
      $("editor-save").disabled = false;
      return;
    }
    state.revision = payload.revision;
    state.segments = payload.segments || [];
    state.drafts = new Map();
    render();
    setStatus("Saved revision " + state.revision);
  }
  document.addEventListener("DOMContentLoaded", () => {
    $("editor-save").addEventListener("click", () => save().catch(() => setError("Save failed")));
    $("editor-reload").addEventListener("click", () => load().catch(() => setError("Load failed")));
    load().catch(() => setError("Load failed"));
  });
})();
