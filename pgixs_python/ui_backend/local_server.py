"""Minimal stdlib local web server for the PGIXS Tab1 UI prototype."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pgixs_python.ui_backend.api import startup_payload, status_payload, workflows_payload
from pgixs_python.ui_backend.preview import preview_error_payload, two_d_info_preview_payload
from pgixs_python.ui_backend.service import UIWorkflowRequest, run_workflow, validate_workflow


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PGIXS Python Tab1</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #d7dee8;
      --text: #1f2933;
      --muted: #66788a;
      --accent: #0f766e;
      --warning: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    button, select, input {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
      min-height: 36px;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      font-weight: 650;
    }
    .app {
      min-height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr) 300px;
      grid-template-rows: minmax(0, 1fr) 180px;
    }
    aside, main, .inspector, .log {
      background: var(--panel);
      border-right: 1px solid var(--line);
      min-width: 0;
    }
    aside { padding: 16px; overflow: auto; }
    main { padding: 16px; overflow: auto; }
    .inspector { padding: 16px; overflow: auto; border-right: 0; }
    .log {
      grid-column: 1 / -1;
      border-top: 1px solid var(--line);
      border-right: 0;
      padding: 12px 16px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
      color: #344054;
    }
    .section { margin-bottom: 18px; }
    .section h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; color: var(--muted); }
    .field { display: grid; gap: 6px; margin-bottom: 10px; }
    .field label { color: var(--muted); font-size: 12px; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
    .tab { padding: 9px 12px; border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: transparent; }
    .tab.active { border-bottom-color: var(--accent); color: var(--accent); font-weight: 650; }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px;
      min-height: 180px;
    }
    .schema-list { display: grid; gap: 8px; }
    .schema-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfdff;
    }
    .schema-item strong { display: block; margin-bottom: 4px; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 650; }
    .muted { color: var(--muted); }
    .warn { color: var(--warning); font-weight: 650; }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 300px minmax(0, 1fr); }
      .inspector { display: none; }
    }
    @media (max-width: 760px) {
      .app { display: block; }
      aside, main, .log { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>PGIXS Python Tab1</h1>
    <div id="runtime" class="muted">Loading backend status...</div>
  </header>
  <div class="app">
    <aside>
      <section class="section">
        <h2>Batch Workflow</h2>
        <div class="field">
          <label for="mode">Primary workflow</label>
          <select id="mode"></select>
        </div>
        <div class="field">
          <label for="batch-path">Batch config JSON</label>
          <input id="batch-path" placeholder="docs/example_batch_workflow.json">
        </div>
        <div class="field">
          <label for="manifest-path">Shared manifest path</label>
          <input id="manifest-path" placeholder="validation_outputs/batch_manifest.jsonl">
        </div>
        <div class="field">
          <label for="log-path">Log / summary folder</label>
          <input id="log-path" placeholder="validation_outputs/batch_logs">
        </div>
        <div class="field">
          <label for="summary-json-path">Batch summary JSON</label>
          <input id="summary-json-path" placeholder="validation_outputs/batch_summary.json">
        </div>
        <div class="field">
          <label for="two-d-info-path">2D-info H5 preview</label>
          <input id="two-d-info-path" placeholder="validation_outputs/sample_2Dinfo.h5">
        </div>
      </section>
      <section class="section">
        <h2>Single Job Defaults</h2>
        <div class="field">
          <label for="job-type">New job type</label>
          <select id="job-type">
            <option>GIXS</option>
            <option>TrXS</option>
            <option>Export 1D</option>
          </select>
        </div>
      </section>
      <section class="section">
        <h2>Schema Controls</h2>
        <div id="schema-list" class="schema-list"></div>
      </section>
    </aside>
    <main>
      <div class="toolbar">
        <button id="validate-batch">Validate Batch</button>
        <button id="run-batch" class="primary">Run Batch</button>
        <button id="load-2d-preview">Load 2D Preview</button>
        <button>Add Job</button>
        <button>Save Batch Config</button>
      </div>
      <div class="tabs">
        <button class="tab active">Batch Queue</button>
        <button class="tab">Run Summary</button>
        <button class="tab">2D Image</button>
        <button class="tab">1D Curve</button>
        <button class="tab">Coverage</button>
        <button class="tab">Report</button>
      </div>
      <section class="panel" id="summary">
        <h2>Batch Queue</h2>
        <p class="muted">No batch has started. Validate Batch checks the job file first; Run Batch starts processing.</p>
        <p class="muted">Single GIXS/TrXS runs will be represented as one-job batch entries.</p>
      </section>
    </main>
    <aside class="inspector">
      <section class="section">
        <h2>Inspector</h2>
        <p id="inspector-text" class="muted">Batch is the primary screen. Select a workflow to inspect config fields.</p>
      </section>
      <section class="section">
        <h2>Warnings</h2>
        <p class="warn">Coverage and detector-gap warnings will stay visible here after each run.</p>
      </section>
    </aside>
    <section class="log" id="log">Starting local UI...</section>
  </div>
  <script>
    function log(message) {
      const node = document.getElementById("log");
      node.textContent += "\\n" + message;
    }

    function buildBatchArgs() {
      const args = [];
      const batchPath = document.getElementById("batch-path").value.trim();
      const manifestPath = document.getElementById("manifest-path").value.trim();
      const logPath = document.getElementById("log-path").value.trim();
      const summaryJsonPath = document.getElementById("summary-json-path").value.trim();
      if (batchPath) args.push("--batch", batchPath);
      if (manifestPath) args.push("--manifest", manifestPath);
      if (logPath) args.push("--log-dir", logPath);
      if (summaryJsonPath) args.push("--summary-json", summaryJsonPath);
      return args;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setSummary(title, body, ok, extraHtml = "") {
      const summary = document.getElementById("summary");
      const cls = ok ? "muted" : "warn";
      summary.innerHTML = `<h2>${title}</h2><p class="${cls}">${body}</p>${extraHtml}`;
    }

    function renderBatchSummaryTable(batchSummary) {
      const jobs = batchSummary?.jobs || [];
      if (!jobs.length) return "";
      const rows = jobs.map((job) => {
        const status = job.ok ? "OK" : `FAIL(${job.exit_code})`;
        const statusClass = job.ok ? "muted" : "warn";
        const logPath = job.log_path ? escapeHtml(job.log_path) : "";
        return `<tr>
          <td>${escapeHtml(job.index)}</td>
          <td>${escapeHtml(job.label)}</td>
          <td>${escapeHtml(job.workflow)}</td>
          <td class="${statusClass}">${escapeHtml(status)}</td>
          <td>${escapeHtml(job.message)}</td>
          <td>${logPath}</td>
        </tr>`;
      }).join("");
      return `<table>
        <thead>
          <tr><th>#</th><th>Label</th><th>Workflow</th><th>Status</th><th>Message</th><th>Log</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
    }

    function renderTwoDPreview(payload) {
      if (!payload.ok) {
        return `<p class="warn">${escapeHtml(payload.message || "2D preview failed")}</p>`;
      }
      const intensity = payload.intensity || {};
      const q = payload.q || {};
      const calibration = payload.calibration || {};
      const rows = [
        ["File", payload.path],
        ["Mode", payload.measurement_mode],
        ["Detector", payload.detector],
        ["Shape", (payload.shape || []).join(" x ")],
        ["Finite fraction", Number(payload.finite_fraction).toFixed(4)],
        ["NaN fraction", Number(payload.nan_fraction).toFixed(4)],
        ["Intensity min/median/max", `${intensity.min} / ${intensity.median} / ${intensity.max}`],
        ["Q min/median/max", `${q.min} / ${q.median} / ${q.max}`],
        ["Beam position", (calibration.beam_position || []).join(", ")],
        ["Wavelength / SDD / Pixel", `${calibration.wavelength} / ${calibration.sdd} / ${calibration.pixel}`],
      ].map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("");
      return `<table><tbody>${rows}</tbody></table>`;
    }

    function validateBatch() {
      const args = buildBatchArgs();
      if (!args.includes("--batch")) {
        setSummary("Batch Validation", "Batch config JSON path is required.", false);
        log("Validation skipped: missing batch config JSON path.");
        return;
      }
      log(`Validating batch: ${args.join(" ")}`);
      fetch("/api/validate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({workflow: "batch", args})
      })
        .then((response) => response.json())
        .then((payload) => {
          const message = payload.message || (payload.ok ? "validation ok" : "validation failed");
          setSummary("Batch Validation", message.replaceAll("\\n", "<br>"), payload.ok);
          log(`Validation ${payload.ok ? "OK" : "FAILED"}: ${message}`);
        })
        .catch((error) => {
          setSummary("Batch Validation", String(error), false);
          log(String(error));
        });
    }

    function runBatch() {
      const args = buildBatchArgs();
      if (!args.includes("--batch")) {
        setSummary("Batch Run", "Batch config JSON path is required.", false);
        log("Run skipped: missing batch config JSON path.");
        return;
      }
      log(`Running batch: ${args.join(" ")}`);
      setSummary("Batch Run", "Batch processing is running. Check the log panel for progress.", true);
      fetch("/api/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({workflow: "batch", args})
      })
        .then((response) => response.json())
        .then((payload) => {
          const message = payload.message || (payload.ok ? "run completed" : "run failed");
          setSummary("Batch Run", escapeHtml(message).replaceAll("\\n", "<br>"), payload.ok, renderBatchSummaryTable(payload.batch_summary));
          log(`Run ${payload.ok ? "OK" : "FAILED"}: ${message}`);
          if (payload.stdout) log(`[stdout]\\n${payload.stdout}`);
          if (payload.stderr) log(`[stderr]\\n${payload.stderr}`);
        })
        .catch((error) => {
          setSummary("Batch Run", String(error), false);
          log(String(error));
        });
    }

    function loadTwoDPreview() {
      const path = document.getElementById("two-d-info-path").value.trim();
      if (!path) {
        setSummary("2D Preview", "2D-info H5 path is required.", false);
        log("2D preview skipped: missing H5 path.");
        return;
      }
      log(`Loading 2D preview: ${path}`);
      fetch("/api/preview/2d", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path})
      })
        .then((response) => response.json())
        .then((payload) => {
          setSummary("2D Preview", payload.ok ? "Loaded compact 2D-info H5 preview." : "Preview failed.", payload.ok, renderTwoDPreview(payload));
          log(`2D preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setSummary("2D Preview", String(error), false);
          log(String(error));
        });
    }

    function renderWorkflowOptions(workflows) {
      const mode = document.getElementById("mode");
      mode.innerHTML = "";
      workflows.forEach((workflow) => {
        const option = document.createElement("option");
        option.value = workflow.workflow;
        option.textContent = workflow.title;
        mode.appendChild(option);
      });
      mode.addEventListener("change", () => renderSchema(workflows.find((item) => item.workflow === mode.value)));
      renderSchema(workflows[0]);
    }

    function renderSchema(workflow) {
      if (!workflow) return;
      const properties = workflow.schema.properties || workflow.schema.$defs?.job?.properties || {};
      const entries = Object.entries(properties).filter(([name]) => !["config"].includes(name)).slice(0, 10);
      const list = document.getElementById("schema-list");
      list.innerHTML = "";
      entries.forEach(([name, schema]) => {
        const item = document.createElement("div");
        item.className = "schema-item";
        item.innerHTML = `<strong>${name}</strong><span class="muted">${schema.description || schema.type || ""}</span>`;
        list.appendChild(item);
      });
      document.getElementById("inspector-text").textContent =
        `${workflow.workflow}: ${Object.keys(properties).length} config fields available.`;
    }

    fetch("/api/startup")
      .then((response) => response.json())
      .then((payload) => {
        const status = payload.status;
        document.getElementById("runtime").textContent =
          `PGIXS ${status.version} | Python ${status.python_version} | ${status.python_ok ? "Python OK" : "Python 3.10 expected"}`;
        renderWorkflowOptions(payload.workflows || []);
        log(`Loaded ${payload.workflows.length} workflow schemas.`);
      })
      .catch((error) => {
        log(String(error));
      });

    document.getElementById("validate-batch").addEventListener("click", validateBatch);
    document.getElementById("run-batch").addEventListener("click", runBatch);
    document.getElementById("load-2d-preview").addEventListener("click", loadTwoDPreview);
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PGIXS Tab1 local web UI scaffold.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local UI server.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local UI server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, int(args.port)), PGIXSRequestHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"PGIXS Tab1 local UI scaffold: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPGIXS local UI server stopped.")
    finally:
        server.server_close()
    return 0


class PGIXSRequestHandler(BaseHTTPRequestHandler):
    server_version = "PGIXSLocalUI/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        if self.path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/status":
            self._send_json(status_payload())
            return
        if self.path == "/api/workflows":
            self._send_json(workflows_payload())
            return
        if self.path == "/api/startup":
            self._send_json(startup_payload())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
        if self.path == "/api/validate":
            self._send_json(validate_payload(self._read_json()))
            return
        if self.path == "/api/run":
            self._send_json(run_payload(self._read_json()))
            return
        if self.path == "/api/preview/2d":
            self._send_json(two_d_preview_payload(self._read_json()))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object.")
        return payload


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    workflow = str(payload.get("workflow", ""))
    args = payload.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        return {
            "workflow": workflow,
            "ok": False,
            "exit_code": 1,
            "message": "args must be a list of strings.",
            "stdout": "",
            "stderr": "",
        }
    request = UIWorkflowRequest(
        workflow=workflow,
        config=payload.get("config"),
        args=list(args),
        label=str(payload.get("label", "")),
    )
    if workflow.strip().lower().replace("_", "-") == "batch":
        dry_run_args = list(request.args)
        if "--dry-run" not in dry_run_args:
            dry_run_args.append("--dry-run")
        request = UIWorkflowRequest(workflow=request.workflow, config=request.config, args=dry_run_args, label=request.label)
        return asdict(run_workflow(request))
    return asdict(validate_workflow(request))


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    workflow = str(payload.get("workflow", ""))
    args = payload.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        return {
            "workflow": workflow,
            "ok": False,
            "exit_code": 1,
            "message": "args must be a list of strings.",
            "stdout": "",
            "stderr": "",
        }
    request = UIWorkflowRequest(
        workflow=workflow,
        config=payload.get("config"),
        args=list(args),
        label=str(payload.get("label", "")),
    )
    result = asdict(run_workflow(request))
    summary = _read_summary_json_arg(args)
    if summary is not None:
        result["batch_summary"] = summary
    return result


def _read_summary_json_arg(args: list[str]) -> dict[str, Any] | None:
    try:
        index = args.index("--summary-json")
    except ValueError:
        return None
    if index + 1 >= len(args):
        return None
    path = Path(args[index + 1])
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def two_d_preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    file_path = str(payload.get("path", "")).strip()
    if not file_path:
        return preview_error_payload(file_path, "path is required.")
    try:
        return two_d_info_preview_payload(file_path)
    except Exception as exc:  # noqa: BLE001 - UI endpoint should return visible failure details.
        return preview_error_payload(file_path, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
