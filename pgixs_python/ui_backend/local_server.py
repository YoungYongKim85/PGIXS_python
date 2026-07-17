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
from pgixs_python.ui_backend.preview import (
    coverage_preview_payload,
    one_d_export_preview_payload,
    preview_error_payload,
    two_d_info_preview_payload,
)
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
    .preview-layout {
      display: grid;
      grid-template-columns: minmax(260px, 420px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .image-preview {
      width: 100%;
      aspect-ratio: 1 / 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f172a;
      image-rendering: pixelated;
    }
    .curve-preview {
      width: 100%;
      min-height: 260px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }
    .curve-path {
      fill: none;
      stroke: var(--accent);
      stroke-width: 2;
    }
    .coverage-path {
      fill: none;
      stroke: var(--warning);
      stroke-width: 1.5;
      stroke-dasharray: 4 4;
    }
    .muted { color: var(--muted); }
    .warn { color: var(--warning); font-weight: 650; }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 300px minmax(0, 1fr); }
      .inspector { display: none; }
    }
    @media (max-width: 760px) {
      .app { display: block; }
      aside, main, .log { border-right: 0; border-bottom: 1px solid var(--line); }
      .preview-layout { grid-template-columns: 1fr; }
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
        <div class="field">
          <label for="one-d-export-path">1D export preview</label>
          <input id="one-d-export-path" placeholder="validation_outputs/sample_1D.csv">
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
        <button id="load-1d-preview">Load 1D Preview</button>
        <button id="load-coverage-preview">Load Coverage</button>
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
      return `<div class="preview-layout">
        <canvas id="two-d-preview-canvas" class="image-preview" width="512" height="512"></canvas>
        <table><tbody>${rows}</tbody></table>
      </div>`;
    }

    function drawTwoDHeatmap(values) {
      const canvas = document.getElementById("two-d-preview-canvas");
      if (!canvas || !values || !values.length || !values[0].length) return;
      const rows = values.length;
      const cols = values[0].length;
      canvas.width = cols;
      canvas.height = rows;
      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(cols, rows);
      const finite = [];
      values.forEach((row) => row.forEach((value) => {
        if (Number.isFinite(value)) finite.push(value);
      }));
      const min = finite.length ? Math.min(...finite) : 0;
      const max = finite.length ? Math.max(...finite) : 1;
      const span = max > min ? max - min : 1;
      for (let y = 0; y < rows; y += 1) {
        for (let x = 0; x < cols; x += 1) {
          const offset = (y * cols + x) * 4;
          const value = values[y][x];
          if (!Number.isFinite(value)) {
            image.data[offset] = 6;
            image.data[offset + 1] = 12;
            image.data[offset + 2] = 24;
            image.data[offset + 3] = 255;
            continue;
          }
          const t = Math.max(0, Math.min(1, (value - min) / span));
          const rgb = heatColor(t);
          image.data[offset] = rgb[0];
          image.data[offset + 1] = rgb[1];
          image.data[offset + 2] = rgb[2];
          image.data[offset + 3] = 255;
        }
      }
      ctx.putImageData(image, 0, 0);
    }

    function heatColor(t) {
      const stops = [
        [24, 36, 96],
        [29, 125, 132],
        [93, 201, 99],
        [252, 231, 77],
      ];
      const scaled = t * (stops.length - 1);
      const index = Math.min(stops.length - 2, Math.floor(scaled));
      const local = scaled - index;
      return stops[index].map((start, channel) => Math.round(start + (stops[index + 1][channel] - start) * local));
    }

    function renderOneDPreview(payload) {
      if (!payload.ok) {
        return `<p class="warn">${escapeHtml(payload.message || "1D preview failed")}</p>`;
      }
      const q = payload.q || {};
      const intensity = payload.intensity || {};
      const coverage = payload.coverage || {};
      const coverageDiagnostics = payload.coverage_diagnostics || {};
      const coverageWarning = coverageDiagnostics.warning_count > 0
        ? `<p class="warn">${escapeHtml(coverageDiagnostics.message)}</p>`
        : (payload.has_coverage ? `<p class="muted">${escapeHtml(coverageDiagnostics.message || "Coverage looks stable.")}</p>` : "");
      const rows = [
        ["File", payload.path],
        ["Points", `${payload.finite_count} finite / ${payload.point_count} total`],
        ["Q min/median/max", `${q.min} / ${q.median} / ${q.max}`],
        ["Intensity min/median/max", `${intensity.min} / ${intensity.median} / ${intensity.max}`],
        ["Std column", payload.has_standard_deviation ? "yes" : "no"],
        ["Coverage column", payload.has_coverage ? `yes, min ${coverage.min}` : "no"],
        ["Low coverage bins", coverageDiagnostics.coverage_low_bins?.length ?? "n/a"],
        ["Coverage drop bins", coverageDiagnostics.coverage_drop_bins?.length ?? "n/a"],
      ].map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("");
      return `${coverageWarning}<div class="preview-layout">
        <svg id="one-d-preview-svg" class="curve-preview" viewBox="0 0 640 300" role="img" aria-label="1D curve preview"></svg>
        <table><tbody>${rows}</tbody></table>
      </div>`;
    }

    function drawOneDCurve(preview) {
      const svg = document.getElementById("one-d-preview-svg");
      if (!svg || !preview) return;
      const q = preview.q || [];
      const intensity = preview.intensity || [];
      const points = q.map((x, index) => [x, intensity[index]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      svg.innerHTML = "";
      const width = 640;
      const height = 300;
      const pad = 34;
      if (!points.length) return;
      const qValues = points.map(([x]) => x);
      const iValues = points.map(([, y]) => y);
      const qMin = Math.min(...qValues);
      const qMax = Math.max(...qValues);
      const iMin = Math.min(...iValues);
      const iMax = Math.max(...iValues);
      const xScale = (value) => pad + ((value - qMin) / Math.max(qMax - qMin, 1e-12)) * (width - pad * 2);
      const yScale = (value) => height - pad - ((value - iMin) / Math.max(iMax - iMin, 1e-12)) * (height - pad * 2);
      const path = points.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${xScale(x).toFixed(2)} ${yScale(y).toFixed(2)}`).join(" ");
      svg.insertAdjacentHTML("beforeend", `<path class="curve-path" d="${path}"></path>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#98a2b3"></line>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#98a2b3"></line>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${pad}" y="22" fill="#66788a" font-size="12">I(q)</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${width - pad - 24}" y="${height - 8}" fill="#66788a" font-size="12">q</text>`);
      if (preview.coverage) drawCoverageCurve(svg, preview.coverage, width, height, pad);
    }

    function drawCoverageCurve(svg, coverage, width, height, pad) {
      const finite = coverage.map((value, index) => [index, value]).filter(([, value]) => Number.isFinite(value));
      if (!finite.length) return;
      const xScale = (index) => pad + (index / Math.max(coverage.length - 1, 1)) * (width - pad * 2);
      const yScale = (value) => height - pad - Math.max(0, Math.min(1, value)) * (height - pad * 2);
      const path = finite.map(([index, value], pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${xScale(index).toFixed(2)} ${yScale(value).toFixed(2)}`).join(" ");
      svg.insertAdjacentHTML("beforeend", `<path class="coverage-path" d="${path}"></path>`);
    }

    function renderCoveragePreview(payload) {
      if (!payload.ok) {
        return `<p class="warn">${escapeHtml(payload.message || "Coverage preview failed")}</p>`;
      }
      const diagnostics = payload.coverage_diagnostics || {};
      const coverage = payload.coverage || {};
      const warning = diagnostics.warning_count > 0
        ? `<p class="warn">${escapeHtml(diagnostics.message)}</p>`
        : `<p class="muted">${escapeHtml(diagnostics.message || "Coverage looks stable.")}</p>`;
      const rows = [
        ["File", payload.path],
        ["Points", `${payload.finite_count} finite / ${payload.point_count} total`],
        ["Coverage min/median/max", `${coverage.min} / ${coverage.median} / ${coverage.max}`],
        ["Low coverage bins", (diagnostics.coverage_low_bins || []).join(", ") || "none"],
        ["Coverage drop bins", (diagnostics.coverage_drop_bins || []).join(", ") || "none"],
      ].map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("");
      return `${warning}<div class="preview-layout">
        <svg id="coverage-preview-svg" class="curve-preview" viewBox="0 0 640 300" role="img" aria-label="Coverage preview"></svg>
        <table><tbody>${rows}</tbody></table>
      </div>`;
    }

    function drawCoveragePreview(payload) {
      const svg = document.getElementById("coverage-preview-svg");
      if (!svg || !payload?.preview) return;
      svg.innerHTML = "";
      const width = 640;
      const height = 300;
      const pad = 34;
      const q = payload.preview.q || [];
      const coverage = payload.preview.coverage || [];
      const points = q.map((x, index) => [x, coverage[index]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      if (!points.length) return;
      const qValues = points.map(([x]) => x);
      const qMin = Math.min(...qValues);
      const qMax = Math.max(...qValues);
      const xScale = (value) => pad + ((value - qMin) / Math.max(qMax - qMin, 1e-12)) * (width - pad * 2);
      const yScale = (value) => height - pad - Math.max(0, Math.min(1, value)) * (height - pad * 2);
      const path = points.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${xScale(x).toFixed(2)} ${yScale(y).toFixed(2)}`).join(" ");
      svg.insertAdjacentHTML("beforeend", `<path class="coverage-path" d="${path}"></path>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#98a2b3"></line>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#98a2b3"></line>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${yScale(0.15).toFixed(2)}" x2="${width - pad}" y2="${yScale(0.15).toFixed(2)}" stroke="#b45309" stroke-dasharray="3 4"></line>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${pad}" y="22" fill="#66788a" font-size="12">coverage</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${width - pad - 24}" y="${height - 8}" fill="#66788a" font-size="12">q</text>`);
      (payload.coverage_diagnostics?.coverage_low_bins || []).forEach((bin) => drawCoverageMarker(svg, bin, coverage.length, width, height, pad, "#b45309"));
      (payload.coverage_diagnostics?.coverage_drop_bins || []).forEach((bin) => drawCoverageMarker(svg, bin, coverage.length, width, height, pad, "#dc2626"));
    }

    function drawCoverageMarker(svg, bin, pointCount, width, height, pad, color) {
      const x = pad + (bin / Math.max(pointCount - 1, 1)) * (width - pad * 2);
      svg.insertAdjacentHTML("beforeend", `<line x1="${x.toFixed(2)}" y1="${pad}" x2="${x.toFixed(2)}" y2="${height - pad}" stroke="${color}" stroke-opacity="0.55"></line>`);
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
          if (payload.ok) requestAnimationFrame(() => drawTwoDHeatmap(payload.preview?.data2d_q));
          log(`2D preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setSummary("2D Preview", String(error), false);
          log(String(error));
        });
    }

    function loadOneDPreview() {
      const path = document.getElementById("one-d-export-path").value.trim();
      if (!path) {
        setSummary("1D Preview", "1D export path is required.", false);
        log("1D preview skipped: missing curve path.");
        return;
      }
      log(`Loading 1D preview: ${path}`);
      fetch("/api/preview/1d", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path})
      })
        .then((response) => response.json())
        .then((payload) => {
          setSummary("1D Preview", payload.ok ? "Loaded 1D export preview." : "Preview failed.", payload.ok, renderOneDPreview(payload));
          if (payload.ok) requestAnimationFrame(() => drawOneDCurve(payload.preview));
          log(`1D preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setSummary("1D Preview", String(error), false);
          log(String(error));
        });
    }

    function loadCoveragePreview() {
      const path = document.getElementById("one-d-export-path").value.trim();
      if (!path) {
        setSummary("Coverage", "1D export path is required.", false);
        log("Coverage preview skipped: missing curve path.");
        return;
      }
      log(`Loading coverage preview: ${path}`);
      fetch("/api/preview/coverage", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path})
      })
        .then((response) => response.json())
        .then((payload) => {
          setSummary("Coverage", payload.ok ? "Loaded coverage diagnostics." : "Coverage preview failed.", payload.ok, renderCoveragePreview(payload));
          if (payload.ok) requestAnimationFrame(() => drawCoveragePreview(payload));
          log(`Coverage preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setSummary("Coverage", String(error), false);
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
    document.getElementById("load-1d-preview").addEventListener("click", loadOneDPreview);
    document.getElementById("load-coverage-preview").addEventListener("click", loadCoveragePreview);
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
        if self.path == "/api/preview/1d":
            self._send_json(one_d_preview_payload(self._read_json()))
            return
        if self.path == "/api/preview/coverage":
            self._send_json(coverage_payload(self._read_json()))
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


def one_d_preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    file_path = str(payload.get("path", "")).strip()
    if not file_path:
        return preview_error_payload(file_path, "path is required.")
    try:
        return one_d_export_preview_payload(file_path)
    except Exception as exc:  # noqa: BLE001 - UI endpoint should return visible failure details.
        return preview_error_payload(file_path, str(exc))


def coverage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    file_path = str(payload.get("path", "")).strip()
    if not file_path:
        return preview_error_payload(file_path, "path is required.")
    try:
        return coverage_preview_payload(file_path)
    except Exception as exc:  # noqa: BLE001 - UI endpoint should return visible failure details.
        return preview_error_payload(file_path, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
