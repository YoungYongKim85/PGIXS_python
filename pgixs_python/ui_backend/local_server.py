"""Minimal stdlib local web server for the PGIXS Tab1 UI prototype."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pgixs_python.ui_backend.api import startup_payload, status_payload, workflows_payload
from pgixs_python.ui_backend.console_session import (
    console_azimuthal_1d,
    console_background_load,
    console_circular_average,
    console_grid_remove,
    console_line_1d,
    console_list_files,
    console_load,
    console_load_info,
    console_load_progress,
    console_normalize,
    console_qmap,
    console_qspace_image,
    console_save,
    console_state,
)
from pgixs_python.ui_backend.preview import (
    coverage_preview_payload,
    one_d_export_preview_payload,
    preview_error_payload,
    two_d_info_preview_payload,
)
from pgixs_python.ui_backend.service import UIWorkflowRequest, run_workflow, validate_workflow
from pgixs_python.workflows.batch_process import load_batch_jobs


_CONSOLE_GIXS_HTML = """
<div class="console-shell">
  <fieldset class="console-panel">
    <legend>Single Data Analysis</legend>

    <div class="c-row">
      <label for="c-dir">Directory</label>
      <input id="c-dir" placeholder="path\\to\\folder" style="flex:1;">
      <button type="button" id="c-browse">Browse</button>
    </div>
    <div class="c-field">
      <label for="c-file">File</label>
      <select id="c-file"><option value="">(browse a directory)</option></select>
    </div>
    <div class="c-field">
      <label for="c-detector">Detector</label>
      <select id="c-detector">
        <option>Eiger4M</option>
        <option>MarCCD</option>
        <option>MX170HS</option>
        <option>PI-CCD</option>
        <option>MAR345</option>
      </select>
    </div>
    <div class="c-row">
      <input type="checkbox" id="c-multishot">
      <label for="c-multishot">Multishot (average this file's series)</label>
    </div>
    <button type="button" id="c-load" class="primary console-load-btn">LOAD DATA</button>

    <div class="c-counters">
      <span>IC: <b id="c-ic">0</b></span>
      <span>BC: <b id="c-bc">0</b></span>
    </div>

    <div class="c-field">
      <label for="c-energy">Energy [keV]</label>
      <input id="c-energy" class="cyan-field" type="number" step="0.001" value="9.5">
    </div>
    <div class="c-field">
      <label for="c-sdd">SDD [mm]</label>
      <input id="c-sdd" class="cyan-field" type="number" step="0.1" value="3000">
    </div>
    <div class="c-field wide">
      <label>Center [y, z]</label>
      <input id="c-beam-y" class="cyan-field" type="number" step="0.1" value="1000">
      <input id="c-beam-z" class="cyan-field" type="number" step="0.1" value="1800">
    </div>
    <div class="c-field">
      <label for="c-footprint">Footprint [mm]</label>
      <input id="c-footprint" type="number" step="0.1" value="1">
    </div>
    <div class="c-row">
      <label for="c-info-dir">Load Info dir</label>
      <input id="c-info-dir" placeholder="path\\to\\folder" style="flex:1;">
      <button type="button" id="c-info-browse">Browse</button>
    </div>
    <div class="c-row">
      <select id="c-info-file" style="flex:1;"><option value="">(browse a directory)</option></select>
      <button type="button" id="c-load-info">Load Info</button>
    </div>
    <button type="button" id="c-grid-remove" class="console-grid-btn console-load-btn">Grid Remove</button>

    <div class="c-row">
      <label for="c-ai-ypix">ai [deg] / y-pos</label>
      <input id="c-ai-ypix" type="number" step="1" value="0" style="width:64px;">
      <button type="button" id="c-ai-zplot" style="margin-left:auto;">z-plot</button>
    </div>
    <div class="c-field wide">
      <label>pixel / degree</label>
      <input id="c-ai-pix" type="number" step="1" value="30">
      <input id="c-ai-deg" class="cyan-field" type="number" step="0.001" value="0.1">
    </div>

    <div class="c-row">
      <label for="c-bkg-dir">Background dir</label>
      <input id="c-bkg-dir" placeholder="path\\to\\folder" style="flex:1;">
      <button type="button" id="c-bkg-browse">Browse</button>
    </div>
    <div class="c-row">
      <select id="c-bkg-file" style="flex:1;"><option value="">(browse a directory)</option></select>
      <button type="button" id="c-load-bkg">Load Bkg.</button>
    </div>
    <div class="c-field">
      <label for="c-plus-data">*Data</label>
      <input id="c-plus-data" type="number" step="0.01" value="1">
    </div>
    <div class="c-field">
      <label for="c-normal-method">Normalization</label>
      <select id="c-normal-method">
        <option>None</option>
        <option>IC</option>
        <option>BC</option>
      </select>
    </div>
    <button type="button" id="c-subtract" style="width:100%; margin-bottom:7px;">BKG sub. &amp; Norm</button>

    <div class="c-row">
      <input type="checkbox" id="c-q-uncertain">
      <label for="c-q-uncertain">q uncertainty</label>
      <select id="c-mapping" style="margin-left:auto;">
        <option>Qxy/Qz</option>
        <option>Qy/Qz</option>
      </select>
    </div>
    <button type="button" id="c-qmap-cal" class="console-qmap-btn">q-Map Cal.</button>
    <p id="c-q-range" class="muted" style="font-size:11px; margin: 6px 0 0;">Q range: not computed yet.</p>

    <div class="c-divider">&mdash; [ 1D Plot Process ] &mdash;</div>
    <div class="c-field">
      <label for="c-line-angle">Line angle</label>
      <input id="c-line-angle" type="number" step="0.1" value="90">
    </div>
    <p class="c-legend">0: &larr;, 90: &uarr;, 180: &rarr;</p>
    <div class="c-row">
      <input type="checkbox" id="c-cakecut">
      <label for="c-cakecut">Cakecut ang (-/+)</label>
      <input id="c-cakecut-minus" type="number" step="0.1" value="-1" style="width:56px;">
      <input id="c-cakecut-plus" type="number" step="0.1" value="1" style="width:56px;">
    </div>
    <div class="c-row">
      <input type="checkbox" id="c-line-median">
      <label for="c-line-median">Median</label>
    </div>
    <div class="c-row">
      <button type="button" id="c-line-position">Position</button>
      <button type="button" id="c-line-draw">Line</button>
      <button type="button" id="c-line-plot">1d Plot</button>
    </div>

    <div class="c-divider">&mdash; [ Azimuthal Process ] &mdash;</div>
    <div class="c-field wide">
      <label>Pixel / Range(-/+)</label>
      <input id="c-azi-pixel" type="number" step="1" value="202">
      <input id="c-azi-range-minus" type="number" step="0.1" value="-1">
    </div>
    <div class="c-field">
      <label for="c-azi-range-plus">Range(+)</label>
      <input id="c-azi-range-plus" type="number" step="0.1" value="1">
    </div>
    <div class="c-field">
      <label for="c-azi-q">Q [1/A]</label>
      <input id="c-azi-q" class="readonly-field" readonly value="0">
    </div>
    <div class="c-row">
      <button type="button" id="c-azi-position">Position</button>
      <select id="c-azi-mean-median">
        <option>Mean</option>
        <option>Median</option>
      </select>
    </div>
    <div class="c-row">
      <button type="button" id="c-azi-draw">Draw</button>
      <button type="button" id="c-azi-plot">Plot</button>
    </div>

    <p id="c-footer" class="console-footer">Python port of the PGIXS GIXS console &middot; Tab1</p>
  </fieldset>

  <div class="console-right">
    <fieldset class="console-panel plot">
      <legend>2D Image Plot</legend>
      <div class="console-image-row">
        <div class="console-axis-y" id="c-axis-y">
          <span class="axis-tick"></span>
          <span class="axis-tick"></span>
          <span class="axis-tick"></span>
        </div>
        <div class="console-plot-col">
          <canvas id="c-canvas" class="console-canvas" width="128" height="128"></canvas>
          <div class="console-axis-x" id="c-axis-x">
            <span class="axis-tick"></span>
            <span class="axis-tick"></span>
            <span class="axis-tick"></span>
          </div>
        </div>
        <div class="console-colorbar-col">
          <span id="c-cbar-max" class="console-colorbar-label">1000</span>
          <canvas id="c-colorbar" class="console-colorbar" width="14" height="256"></canvas>
          <span id="c-cbar-min" class="console-colorbar-label">0</span>
        </div>
      </div>
      <p id="c-status" class="muted" style="margin: 8px 0 0;"></p>
    </fieldset>
    <fieldset class="console-panel controller">
      <legend>Image Controller</legend>
      <div class="console-controller-grid">
        <div class="c-field">
          <label for="c-color-scale">ColorScale</label>
          <select id="c-color-scale"><option>Linear</option><option>Log</option></select>
        </div>
        <div class="c-field">
          <label for="c-colormap">Colormap</label>
          <select id="c-colormap"><option value="heat">heat</option><option value="gray">gray</option><option value="hot">hot</option><option value="jet">jet</option><option value="turbo">turbo</option><option value="viridis">viridis</option><option value="inferno">inferno</option><option value="magma">magma</option><option value="parula">parula</option><option value="ocean">ocean</option><option value="rwb">rwb</option><option value="rkb">rkb</option></select>
        </div>
        <div class="c-field">
          <label>Contrast min/max</label>
          <div class="range-pair"><input id="c-cmin" type="number" value="0"><input id="c-cmax" type="number" value="1000"></div>
        </div>
        <div class="c-field">
          <label for="c-map-format">Map format</label>
          <select id="c-map-format"><option>Pixel</option><option>Qxy/Qz</option><option>Qy/Qz</option></select>
        </div>
        <div class="c-field">
          <label>Y-Range</label>
          <div class="range-pair"><input id="c-yrange-min" type="number" value="1"><input id="c-yrange-max" type="number" value="2000"></div>
        </div>
        <div class="c-field">
          <label>Z-Range</label>
          <div class="range-pair"><input id="c-zrange-min" type="number" value="1"><input id="c-zrange-max" type="number" value="2000"></div>
        </div>
        <div class="c-field">
          <label for="c-font-sz">Font Sz</label>
          <input id="c-font-sz" type="number" value="11" min="8" max="22" step="1">
        </div>
      </div>
      <div class="c-row" style="margin-top:8px;">
        <button type="button" id="c-update">Update</button>
        <button type="button" id="c-reset-zoom">Reset Zoom</button>
        <label for="c-save-format" style="margin-left:auto;">Format</label>
        <select id="c-save-format"><option value="png">PNG</option><option value="jpg">JPG</option></select>
        <button type="button" id="c-save" class="save">Save Img</button>
      </div>
    </fieldset>
    <fieldset class="console-panel">
      <legend>1D Result</legend>
      <svg id="c-curve-svg" class="console-curve-svg" viewBox="0 0 640 260" role="img" aria-label="1D curve result"></svg>
      <p id="c-curve-status" class="muted" style="margin: 6px 0 0;"></p>
    </fieldset>
  </div>
</div>
"""

_CONSOLE_TRXS_HTML = """
<div class="console-shell">
  <fieldset class="console-panel">
    <legend>Single Data Analysis</legend>

    <div class="c-row">
      <label for="t-dir">Directory</label>
      <input id="t-dir" placeholder="path\\to\\folder" style="flex:1;">
      <button type="button" id="t-browse">Browse</button>
    </div>
    <div class="c-field">
      <label for="t-file">File</label>
      <select id="t-file"><option value="">(browse a directory)</option></select>
    </div>
    <div class="c-field">
      <label for="t-detector">Detector</label>
      <select id="t-detector">
        <option>Eiger4M</option>
        <option>MarCCD</option>
        <option>MX170HS</option>
        <option>PI_CCD</option>
      </select>
    </div>
    <div class="c-row">
      <input type="checkbox" id="t-multishot">
      <label for="t-multishot">Multishot (average this file's series)</label>
    </div>
    <button type="button" id="t-load" class="primary console-load-btn">LOAD DATA</button>

    <div class="c-counters">
      <span>IC: <b id="t-ic">0</b></span>
      <span>BC: <b id="t-bc">0</b></span>
    </div>

    <div class="c-field">
      <label for="t-energy">Energy [keV]</label>
      <input id="t-energy" class="cyan-field" type="number" step="0.001" value="9.5">
    </div>
    <div class="c-row">
      <label for="t-info-dir">Load Info dir</label>
      <input id="t-info-dir" placeholder="path\\to\\folder" style="flex:1;">
      <button type="button" id="t-info-browse">Browse</button>
    </div>
    <div class="c-row">
      <select id="t-info-file" style="flex:1;"><option value="">(browse a directory)</option></select>
      <button type="button" id="t-load-info">Load Info</button>
    </div>
    <div class="c-field">
      <label for="t-sdd">SDD [mm]</label>
      <input id="t-sdd" class="cyan-field" type="number" step="0.1" value="3000">
    </div>
    <div class="c-field wide">
      <label>Center [y, z]</label>
      <input id="t-beam-y" class="cyan-field" type="number" step="0.1" value="1000">
      <input id="t-beam-z" class="cyan-field" type="number" step="0.1" value="1800">
    </div>

    <div class="c-row">
      <label for="t-bkg-dir">Bkg dir</label>
      <input id="t-bkg-dir" placeholder="path\\to\\folder" style="flex:1;">
      <button type="button" id="t-bkg-browse">Browse</button>
    </div>
    <div class="c-row">
      <select id="t-bkg-file" style="flex:1;"><option value="">(browse a directory)</option></select>
      <button type="button" id="t-load-bkg">Load Bkg.</button>
    </div>
    <div class="c-field">
      <label for="t-plus-data">Multi</label>
      <input id="t-plus-data" type="number" step="0.01" value="1">
    </div>
    <div class="c-field">
      <label for="t-normal-method">Normal</label>
      <select id="t-normal-method">
        <option>None</option>
        <option>IC</option>
        <option>BC</option>
      </select>
    </div>
    <button type="button" id="t-subtract" style="width:100%; margin-bottom:7px;">BKG sub. &amp; Norm</button>
    <button type="button" id="t-grid-remove" class="console-grid-btn console-load-btn">Grid Remove</button>
    <button type="button" id="t-qmap-cal" class="console-qmap-btn">q-Map Cal.</button>
    <p id="t-q-range" class="muted" style="font-size:11px; margin: 6px 0 0;">Q range: not computed yet.</p>

    <div class="c-divider">&mdash; [ 1D Plot Process ] &mdash;</div>
    <div class="c-field">
      <label for="t-min-coverage">Min coverage</label>
      <input id="t-min-coverage" type="number" step="0.05" min="0" max="1" value="0.80">
    </div>
    <div class="c-row">
      <input type="checkbox" id="t-median">
      <label for="t-median">Median</label>
      <input type="checkbox" id="t-std">
      <label for="t-std">Std</label>
    </div>
    <div class="c-field wide">
      <label>Ang. Avg (min/max)</label>
      <input id="t-ang-min" type="number" step="1" value="0">
      <input id="t-ang-max" type="number" step="1" value="360">
    </div>
    <button type="button" id="t-circular-plot" style="width:100%; margin-bottom:7px;">Plot (Circular Average)</button>

    <div class="c-field">
      <label for="t-line-angle">1D Line Ang</label>
      <input id="t-line-angle" type="number" step="0.1" value="0">
    </div>
    <div class="c-row">
      <input type="checkbox" id="t-line-median" hidden>
      <button type="button" id="t-line-draw">Line</button>
      <button type="button" id="t-line-plot">Plot</button>
    </div>

    <div class="c-divider">&mdash; [ Azimuthal Process ] &mdash;</div>
    <div class="c-field wide">
      <label>Pixel / Range(-/+)</label>
      <input id="t-azi-pixel" type="number" step="1" value="202">
      <input id="t-azi-range-minus" type="number" step="0.1" value="-1">
    </div>
    <div class="c-field">
      <label for="t-azi-range-plus">Range(+)</label>
      <input id="t-azi-range-plus" type="number" step="0.1" value="1">
    </div>
    <div class="c-field">
      <label for="t-azi-q">Q [1/A]</label>
      <input id="t-azi-q" class="readonly-field" readonly value="0">
    </div>
    <div class="c-row">
      <button type="button" id="t-azi-position">Point Find</button>
      <select id="t-azi-mean-median">
        <option>Mean</option>
        <option>Median</option>
      </select>
    </div>
    <div class="c-row">
      <button type="button" id="t-azi-draw">Draw</button>
      <button type="button" id="t-azi-plot">Plot</button>
    </div>

    <p id="t-footer" class="console-footer">Python port of the PGIXS TrXS console &middot; Tab1</p>
  </fieldset>

  <div class="console-right">
    <fieldset class="console-panel plot">
      <legend>2D Plot Viewer</legend>
      <div class="console-image-row">
        <div class="console-axis-y" id="t-axis-y">
          <span class="axis-tick"></span>
          <span class="axis-tick"></span>
          <span class="axis-tick"></span>
        </div>
        <div class="console-plot-col">
          <canvas id="t-canvas" class="console-canvas" width="128" height="128"></canvas>
          <div class="console-axis-x" id="t-axis-x">
            <span class="axis-tick"></span>
            <span class="axis-tick"></span>
            <span class="axis-tick"></span>
          </div>
        </div>
        <div class="console-colorbar-col">
          <span id="t-cbar-max" class="console-colorbar-label">1000</span>
          <canvas id="t-colorbar" class="console-colorbar" width="14" height="256"></canvas>
          <span id="t-cbar-min" class="console-colorbar-label">0</span>
        </div>
      </div>
      <p id="t-status" class="muted" style="margin: 8px 0 0;"></p>
    </fieldset>
    <fieldset class="console-panel controller">
      <legend>Image Controller</legend>
      <div class="console-controller-grid">
        <div class="c-field">
          <label for="t-color-scale">ColorScale</label>
          <select id="t-color-scale"><option>Linear</option><option>Log</option></select>
        </div>
        <div class="c-field">
          <label for="t-colormap">Colormap</label>
          <select id="t-colormap"><option value="heat">heat</option><option value="gray">gray</option><option value="hot">hot</option><option value="jet">jet</option><option value="turbo">turbo</option><option value="viridis">viridis</option><option value="inferno">inferno</option><option value="magma">magma</option><option value="parula">parula</option><option value="ocean">ocean</option><option value="rwb">rwb</option><option value="rkb">rkb</option></select>
        </div>
        <div class="c-field">
          <label>Min / Max</label>
          <div class="range-pair"><input id="t-cmin" type="number" value="0"><input id="t-cmax" type="number" value="1000"></div>
        </div>
        <div class="c-field">
          <label for="t-map-format">Map format</label>
          <select id="t-map-format"><option>Pixel</option><option>Qy/Qz</option></select>
        </div>
        <div class="c-field">
          <label>Y-Range</label>
          <div class="range-pair"><input id="t-yrange-min" type="number" value="1"><input id="t-yrange-max" type="number" value="2000"></div>
        </div>
        <div class="c-field">
          <label>Z-Range</label>
          <div class="range-pair"><input id="t-zrange-min" type="number" value="1"><input id="t-zrange-max" type="number" value="2000"></div>
        </div>
        <div class="c-field">
          <label for="t-font-sz">Font Sz</label>
          <input id="t-font-sz" type="number" value="11" min="8" max="22" step="1">
        </div>
      </div>
      <div class="c-row" style="margin-top:8px;">
        <button type="button" id="t-update">Update</button>
        <button type="button" id="t-reset-zoom">Reset Zoom</button>
        <label for="t-save-format" style="margin-left:auto;">Format</label>
        <select id="t-save-format"><option value="png">PNG</option><option value="jpg">JPG</option></select>
        <button type="button" id="t-save" class="save">Save Image</button>
      </div>
    </fieldset>
    <fieldset class="console-panel">
      <legend>1D Result</legend>
      <svg id="t-curve-svg" class="console-curve-svg" viewBox="0 0 640 260" role="img" aria-label="1D curve result"></svg>
      <p id="t-curve-status" class="muted" style="margin: 6px 0 0;"></p>
    </fieldset>
  </div>
</div>
"""


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PGIXS Python Tab1</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f2f2f2;
      --panel: #fafafa;
      --panel-2: #ffffff;
      --line: #adadad;
      --line-soft: #d8d8d8;
      --text: #1a1a1a;
      --muted: #5c5c5c;
      --text-dim: #808080;
      --accent: #1a4d80;
      --accent-dim: #6b86a3;
      --on-accent: #ffffff;
      --data: #185fa5;
      --warning: #b45309;
      --danger: #a01010;
      --editable-bg: #ccffff;
      --readonly-bg: #e5e5e5;
      --save-bg: #e5f2ff;
      --success: #1a7a3c;
      --font-sans: "Segoe UI", "Helvetica Neue", -apple-system, "Noto Sans", Arial, sans-serif;
      --font-mono: ui-monospace, "Cascadia Mono", "Cascadia Code", Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font-sans);
      color: var(--text);
      background: var(--bg);
    }
    header {
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 14px; font-weight: 700; }
    #runtime.status-chip { font-family: var(--font-mono); font-size: 11px; color: var(--muted); }
    .workspace-switch {
      display: flex;
      gap: 0;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 0 20px;
    }
    .workspace-tab {
      padding: 9px 18px;
      border: 1px solid var(--line);
      border-bottom: none;
      border-radius: 3px 3px 0 0;
      background: #e8e8e8;
      color: var(--muted);
      font-size: 12.5px;
      font-weight: 600;
      margin: 6px 4px 0 0;
    }
    .workspace-tab.active { background: var(--bg); color: var(--text); }
    .workspace-tab[disabled] { opacity: 0.5; cursor: not-allowed; }
    .workspace[hidden] { display: none; }
    button, select, input {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: var(--panel-2);
      color: var(--text);
      padding: 7px 9px;
      min-height: 32px;
    }
    button { font-weight: 600; font-size: 12.5px; cursor: pointer; background: #e8e8e8; }
    button:hover { background: #dedede; }
    button.primary {
      background: var(--accent);
      color: var(--on-accent);
      border-color: var(--accent);
      font-weight: 700;
    }
    button.primary:hover { background: #143f69; }
    button.save { background: var(--save-bg); }
    .app {
      min-height: calc(100vh - 48px);
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr) 300px;
    }
    aside, main, .inspector {
      background: var(--panel);
      border-right: 1px solid var(--line-soft);
      min-width: 0;
    }
    aside { padding: 16px; overflow: auto; }
    main { padding: 16px; overflow: auto; background: var(--bg); }
    .inspector { padding: 16px; overflow: auto; border-right: 0; }
    .log {
      border-top: 1px solid var(--line);
      padding: 10px 20px;
      max-height: 160px;
      overflow: auto;
      background: #101418;
      font-family: var(--font-mono);
      font-size: 11.5px;
      line-height: 1.7;
      color: #cfd6e0;
    }
    .log .t { color: #7c8899; }
    .section { margin-bottom: 18px; }
    .section h2 {
      margin: 0 0 10px;
      padding-bottom: 8px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--text-dim);
      border-bottom: 1px solid var(--line-soft);
    }
    .field { display: grid; gap: 6px; margin-bottom: 10px; }
    .field label { color: var(--muted); font-size: 12px; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
    .tab {
      padding: 9px 12px;
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      background: transparent;
      font-family: var(--font-mono);
      font-size: 11.5px;
      letter-spacing: 0.03em;
      color: var(--muted);
      font-weight: 500;
    }
    .tab .count { color: var(--text-dim); }
    .tab.active { border-bottom-color: var(--accent); color: var(--accent); font-weight: 650; }
    .tab.active .count { color: var(--accent-dim); }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-height: 180px;
    }
    .panel[hidden] { display: none; }
    .schema-list { display: grid; gap: 8px; }
    .schema-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: var(--panel-2);
    }
    .schema-item strong { display: block; margin-bottom: 4px; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line-soft);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--text-dim);
      font-weight: 400;
      font-family: var(--font-mono);
      font-size: 10.5px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border-bottom: 1px solid var(--line);
    }
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
      background: var(--panel-2);
      image-rendering: pixelated;
    }
    .curve-preview {
      width: 100%;
      min-height: 260px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
    }
    .curve-path {
      fill: none;
      stroke: var(--data);
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
    .ok-mark { color: var(--success); font-weight: 700; }
    #batch-jobs-table { width: 100%; }
    #batch-jobs-table td { padding: 6px; }
    #batch-jobs-table input, #batch-jobs-table select { width: 100%; min-height: 32px; padding: 6px 8px; }
    #batch-jobs-table select { font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; }
    #batch-jobs-table input.mono-input { font-family: var(--font-mono); font-size: 11.5px; }
    #batch-jobs-table button { padding: 4px 8px; min-height: 28px; }
    .row-remove {
      font-family: var(--font-mono);
      font-size: 11px;
      letter-spacing: 0.04em;
      color: var(--text-dim);
      background: none;
      border: none;
      min-height: auto;
    }
    .row-remove:hover { color: var(--danger); border-color: transparent; }
    #batch-status .ok-mark { margin-right: 4px; }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 300px minmax(0, 1fr); }
      .inspector { display: none; }
    }
    @media (max-width: 760px) {
      .app { display: block; }
      aside, main { border-right: 0; border-bottom: 1px solid var(--line); }
      .preview-layout { grid-template-columns: 1fr; }
    }

    /* GIXS console */
    .console-shell {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
    }
    .console-right { display: grid; gap: 12px; min-width: 0; }
    fieldset.console-panel {
      border: 1px solid var(--line);
      border-radius: 3px;
      background: var(--panel);
      padding: 10px 12px 12px;
      margin: 0;
    }
    fieldset.console-panel.plot { background: var(--panel-2); }
    .console-panel legend { padding: 0 6px; font-size: 13px; font-weight: 700; color: var(--text); }
    .console-panel.controller legend { font-size: 12px; }
    .c-field { display: grid; grid-template-columns: 96px 1fr; align-items: center; gap: 6px; margin-bottom: 7px; }
    .c-field.wide { grid-template-columns: 96px 1fr 1fr; }
    .c-field label { font-size: 11.5px; color: var(--text); }
    .c-field input, .c-field select { width: 100%; min-height: 28px; padding: 4px 6px; font-size: 12px; }
    .c-field input.readonly-field { background: var(--readonly-bg); color: var(--muted); }
    .c-field input.cyan-field { background: var(--editable-bg); }
    .c-row { display: flex; gap: 6px; align-items: center; margin-bottom: 7px; }
    .c-row label { font-size: 11.5px; white-space: nowrap; flex: none; }
    .c-row button { flex: none; }
    .c-divider { text-align: center; font-size: 11.5px; font-weight: 700; color: var(--text); margin: 12px 0 8px; }
    .c-legend { font-size: 10.5px; color: var(--muted); text-align: right; }
    .c-counters { display: flex; gap: 14px; font-size: 12px; margin-bottom: 7px; }
    .c-counters b { font-weight: 700; }
    .console-load-btn { width: 100%; margin-bottom: 7px; }
    .console-qmap-btn { width: 100%; }
    .console-qmap-btn.ok, .console-grid-btn.ok { background: #cdeccd; border-color: var(--success); }
    .console-canvas {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      border: 1px solid var(--line);
      image-rendering: pixelated;
      cursor: crosshair;
      background: #ffffff;
    }
    .console-image-row {
      display: flex;
      align-items: stretch;
      gap: 6px;
      width: 100%;
      max-width: 640px;
      margin: 0 auto;
    }
    .console-plot-col {
      display: flex;
      flex-direction: column;
      flex: 1 1 auto;
      min-width: 0;
    }
    .console-plot-col .console-canvas {
      flex: 1 1 auto;
      min-width: 0;
    }
    .console-axis-y {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: flex-end;
      flex: none;
      padding-right: 3px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--muted);
    }
    .console-axis-x {
      display: flex;
      justify-content: space-between;
      margin-top: 2px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--muted);
    }
    .console-colorbar-col {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      flex: none;
      width: 34px;
    }
    .console-colorbar {
      flex: 1 1 auto;
      width: 14px;
      min-height: 0;
      border: 1px solid var(--line);
      image-rendering: pixelated;
    }
    .console-colorbar-label {
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--muted);
      flex: none;
    }
    .console-controller-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      align-items: end;
    }
    .console-controller-grid .c-field { grid-template-columns: 1fr; gap: 2px; margin-bottom: 0; }
    .console-controller-grid .c-field label { font-size: 10.5px; color: var(--muted); }
    .console-controller-grid .range-pair { display: flex; gap: 4px; }
    .console-curve-svg { width: 100%; min-height: 200px; background: var(--panel-2); border: 1px solid var(--line); }
    .console-curve-svg .curve-path { fill: none; stroke: var(--data); stroke-width: 1.75; }
    .console-curve-svg .coverage-path { fill: none; stroke: var(--warning); stroke-width: 1.25; stroke-dasharray: 4 4; }
    .console-footer { text-align: center; padding: 8px 0 0; font-size: 10.5px; color: var(--text-dim); }
  </style>
</head>
<body>
  <header>
    <h1>PGIXS Python &mdash; Tab1</h1>
    <div id="runtime" class="status-chip">Loading backend status...</div>
  </header>
  <nav class="workspace-switch">
    <button type="button" class="workspace-tab active" id="workspace-tab-console-gixs" data-workspace="console-gixs">Console (GIXS)</button>
    <button type="button" class="workspace-tab" id="workspace-tab-console-trxs" data-workspace="console-trxs">Console (TrXS)</button>
    <button type="button" class="workspace-tab" id="workspace-tab-batch" data-workspace="batch">Batch</button>
  </nav>
  <div id="workspace-console-gixs" class="workspace">
    __CONSOLE_GIXS_HTML__
  </div>
  <div id="workspace-console-trxs" class="workspace" hidden>
    __CONSOLE_TRXS_HTML__
  </div>
  <div id="workspace-batch" class="workspace" hidden>
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
        <button id="add-job">Add Job</button>
        <button id="save-batch-config">Save Batch Config</button>
        <button id="load-batch-config">Load Batch Config</button>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="batch">Batch Queue <span class="count" id="batch-tab-count"></span></button>
        <button class="tab" data-tab="summary">Run Summary</button>
        <button class="tab" data-tab="image">2D Image</button>
        <button class="tab" data-tab="curve">1D Curve</button>
        <button class="tab" data-tab="coverage">Coverage</button>
        <button class="tab" data-tab="report">Report</button>
      </div>
      <section class="panel tab-panel" id="panel-batch" data-panel="batch">
        <h2>Batch Queue</h2>
        <p class="muted">Add jobs below, then Save Batch Config before Validate/Run. Load Batch Config reads the path above into this table.</p>
        <table id="batch-jobs-table">
          <thead>
            <tr><th>Label</th><th>Workflow</th><th>Config path</th><th>Args</th><th></th></tr>
          </thead>
          <tbody id="batch-jobs-body"></tbody>
        </table>
        <p id="batch-jobs-empty" class="muted">No jobs yet. Use Add Job to create the first row.</p>
        <p id="batch-status" class="muted"></p>
      </section>
      <section class="panel tab-panel" id="panel-summary" data-panel="summary" hidden>
        <h2>Run Summary</h2>
        <p class="muted">Run Batch results will appear here.</p>
      </section>
      <section class="panel tab-panel" id="panel-image" data-panel="image" hidden>
        <h2>2D Image</h2>
        <p class="muted">Load a 2D-info H5 preview to inspect the compact heatmap.</p>
      </section>
      <section class="panel tab-panel" id="panel-curve" data-panel="curve" hidden>
        <h2>1D Curve</h2>
        <p class="muted">Load a 1D CSV/DAT preview to inspect q-intensity data.</p>
      </section>
      <section class="panel tab-panel" id="panel-coverage" data-panel="coverage" hidden>
        <h2>Coverage</h2>
        <p class="muted">Load Coverage to inspect detector-gap affected bins.</p>
      </section>
      <section class="panel tab-panel" id="panel-report" data-panel="report" hidden>
        <h2>Report</h2>
        <p class="muted">Report preview and links will appear here.</p>
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
  </div>
  </div>
  <section class="log" id="log"><div><span class="t">--:--:--</span> <span class="msg">starting local ui...</span></div></section>
  <script>
    function log(message) {
      const node = document.getElementById("log");
      const time = new Date().toLocaleTimeString("en-GB", {hour12: false});
      const body = escapeHtml(String(message)).replaceAll("\\n", "<br>");
      node.insertAdjacentHTML("beforeend", `<div><span class="t">${time}</span> <span class="msg">${body}</span></div>`);
      node.scrollTop = node.scrollHeight;
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

    function activateTab(tabName) {
      document.querySelectorAll(".tab").forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.tab === tabName);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.dataset.panel !== tabName;
      });
    }

    function setPanel(tabName, title, body, ok, extraHtml = "") {
      const summary = document.getElementById(`panel-${tabName}`);
      const cls = ok ? "muted" : "warn";
      summary.innerHTML = `<h2>${title}</h2><p class="${cls}">${body}</p>${extraHtml}`;
      activateTab(tabName);
    }

    let batchJobs = [];

    const BATCH_WORKFLOW_OPTIONS = ["gixs", "trxs", "export-1d"];
    const JOB_TYPE_TO_WORKFLOW = {GIXS: "gixs", TrXS: "trxs", "Export 1D": "export-1d"};

    function setBatchStatus(message, ok) {
      const node = document.getElementById("batch-status");
      node.className = ok ? "muted" : "warn";
      const prefix = ok ? '<span class="ok-mark">&#10003;</span> ' : "";
      node.innerHTML = prefix + escapeHtml(message).replaceAll("\\n", "<br>");
      activateTab("batch");
    }

    function renderBatchTable() {
      const body = document.getElementById("batch-jobs-body");
      const empty = document.getElementById("batch-jobs-empty");
      empty.hidden = batchJobs.length > 0;
      body.innerHTML = batchJobs.map((job, index) => `
        <tr>
          <td><input data-field="label" data-index="${index}" value="${escapeHtml(job.label)}"></td>
          <td>
            <select data-field="workflow" data-index="${index}">
              ${BATCH_WORKFLOW_OPTIONS.map((option) => `<option value="${option}" ${option === job.workflow ? "selected" : ""}>${option}</option>`).join("")}
            </select>
          </td>
          <td><input class="mono-input" data-field="config" data-index="${index}" value="${escapeHtml(job.config || "")}"></td>
          <td><input class="mono-input" data-field="args" data-index="${index}" value="${escapeHtml((job.args || []).join(" "))}"></td>
          <td><button type="button" class="row-remove" data-remove="${index}">remove</button></td>
        </tr>
      `).join("");
      const countNode = document.getElementById("batch-tab-count");
      if (countNode) countNode.textContent = batchJobs.length ? `\xb7 ${batchJobs.length}` : "";
    }

    function addJob() {
      const jobType = document.getElementById("job-type").value;
      const workflow = JOB_TYPE_TO_WORKFLOW[jobType] || "gixs";
      batchJobs.push({label: `job-${batchJobs.length + 1}`, workflow, config: "", args: []});
      renderBatchTable();
      activateTab("batch");
    }

    function saveBatchConfig() {
      const path = document.getElementById("batch-path").value.trim();
      if (!path) {
        setBatchStatus("Batch config JSON path is required to save.", false);
        log("Save skipped: missing batch config JSON path.");
        return Promise.resolve({ok: false});
      }
      if (!batchJobs.length) {
        setBatchStatus("Add at least one job before saving.", false);
        log("Save skipped: no jobs in the table.");
        return Promise.resolve({ok: false});
      }
      const jobs = batchJobs.map((job) => ({
        label: job.label,
        workflow: job.workflow,
        config: job.config || null,
        args: job.args || [],
      }));
      log(`Saving batch config: ${path}`);
      return fetch("/api/batch/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path, jobs})
      })
        .then((response) => response.json())
        .then((payload) => {
          const message = payload.ok
            ? `Saved ${payload.job_count} job(s) to ${payload.path}.`
            : (payload.message || "Save failed.");
          setBatchStatus(message, payload.ok);
          log(`Save ${payload.ok ? "OK" : "FAILED"}: ${message}`);
          return payload;
        })
        .catch((error) => {
          setBatchStatus(String(error), false);
          log(String(error));
          return {ok: false};
        });
    }

    function loadBatchConfig() {
      const path = document.getElementById("batch-path").value.trim();
      if (!path) {
        setBatchStatus("Batch config JSON path is required to load.", false);
        log("Load skipped: missing batch config JSON path.");
        return;
      }
      log(`Loading batch config: ${path}`);
      fetch(`/api/batch/load?path=${encodeURIComponent(path)}`)
        .then((response) => response.json())
        .then((payload) => {
          if (!payload.ok) {
            setBatchStatus(payload.message || "Load failed.", false);
            log(`Load FAILED: ${payload.message || path}`);
            return;
          }
          batchJobs = (payload.jobs || []).map((job) => ({
            label: job.label || "",
            workflow: job.workflow || "gixs",
            config: job.config || "",
            args: job.args || [],
          }));
          renderBatchTable();
          setBatchStatus(`Loaded ${batchJobs.length} job(s) from ${payload.path}.`, true);
          log(`Load OK: ${payload.path}`);
        })
        .catch((error) => {
          setBatchStatus(String(error), false);
          log(String(error));
        });
    }

    function ensureBatchSaved() {
      if (!batchJobs.length) return Promise.resolve(true);
      return saveBatchConfig().then((payload) => Boolean(payload && payload.ok));
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
        [8, 12, 20],
        [21, 92, 106],
        [47, 217, 196],
        [242, 179, 61],
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
      svg.insertAdjacentHTML("beforeend", `<text x="${pad}" y="22" fill="#8b96a8" font-size="12">I(q)</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${width - pad - 24}" y="${height - 8}" fill="#8b96a8" font-size="12">q</text>`);
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
      svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${yScale(0.15).toFixed(2)}" x2="${width - pad}" y2="${yScale(0.15).toFixed(2)}" stroke="#f2994a" stroke-dasharray="3 4"></line>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${pad}" y="22" fill="#8b96a8" font-size="12">coverage</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${width - pad - 24}" y="${height - 8}" fill="#8b96a8" font-size="12">q</text>`);
      (payload.coverage_diagnostics?.coverage_low_bins || []).forEach((bin) => drawCoverageMarker(svg, bin, coverage.length, width, height, pad, "#f2994a"));
      (payload.coverage_diagnostics?.coverage_drop_bins || []).forEach((bin) => drawCoverageMarker(svg, bin, coverage.length, width, height, pad, "#f2545b"));
    }

    function drawCoverageMarker(svg, bin, pointCount, width, height, pad, color) {
      const x = pad + (bin / Math.max(pointCount - 1, 1)) * (width - pad * 2);
      svg.insertAdjacentHTML("beforeend", `<line x1="${x.toFixed(2)}" y1="${pad}" x2="${x.toFixed(2)}" y2="${height - pad}" stroke="${color}" stroke-opacity="0.55"></line>`);
    }

    function validateBatch() {
      ensureBatchSaved().then((saved) => {
        if (!saved) return;
        const args = buildBatchArgs();
        if (!args.includes("--batch")) {
          setBatchStatus("Batch config JSON path is required.", false);
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
            setBatchStatus(message, payload.ok);
            log(`Validation ${payload.ok ? "OK" : "FAILED"}: ${message}`);
          })
          .catch((error) => {
            setBatchStatus(String(error), false);
            log(String(error));
          });
      });
    }

    function runBatch() {
      ensureBatchSaved().then((saved) => {
        if (!saved) return;
        const args = buildBatchArgs();
        if (!args.includes("--batch")) {
          setPanel("summary", "Batch Run", "Batch config JSON path is required.", false);
          log("Run skipped: missing batch config JSON path.");
          return;
        }
        log(`Running batch: ${args.join(" ")}`);
        setPanel("summary", "Batch Run", "Batch processing is running. Check the log panel for progress.", true);
        fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({workflow: "batch", args})
        })
          .then((response) => response.json())
          .then((payload) => {
            const message = payload.message || (payload.ok ? "run completed" : "run failed");
            setPanel("summary", "Batch Run", escapeHtml(message).replaceAll("\\n", "<br>"), payload.ok, renderBatchSummaryTable(payload.batch_summary));
            log(`Run ${payload.ok ? "OK" : "FAILED"}: ${message}`);
            if (payload.stdout) log(`[stdout]\\n${payload.stdout}`);
            if (payload.stderr) log(`[stderr]\\n${payload.stderr}`);
          })
          .catch((error) => {
            setPanel("summary", "Batch Run", String(error), false);
            log(String(error));
          });
      });
    }

    function loadTwoDPreview() {
      const path = document.getElementById("two-d-info-path").value.trim();
      if (!path) {
        setPanel("image", "2D Preview", "2D-info H5 path is required.", false);
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
          setPanel("image", "2D Preview", payload.ok ? "Loaded compact 2D-info H5 preview." : "Preview failed.", payload.ok, renderTwoDPreview(payload));
          if (payload.ok) requestAnimationFrame(() => drawTwoDHeatmap(payload.preview?.data2d_q));
          log(`2D preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setPanel("image", "2D Preview", String(error), false);
          log(String(error));
        });
    }

    function loadOneDPreview() {
      const path = document.getElementById("one-d-export-path").value.trim();
      if (!path) {
        setPanel("curve", "1D Preview", "1D export path is required.", false);
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
          setPanel("curve", "1D Preview", payload.ok ? "Loaded 1D export preview." : "Preview failed.", payload.ok, renderOneDPreview(payload));
          if (payload.ok) requestAnimationFrame(() => drawOneDCurve(payload.preview));
          log(`1D preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setPanel("curve", "1D Preview", String(error), false);
          log(String(error));
        });
    }

    function loadCoveragePreview() {
      const path = document.getElementById("one-d-export-path").value.trim();
      if (!path) {
        setPanel("coverage", "Coverage", "1D export path is required.", false);
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
          setPanel("coverage", "Coverage", payload.ok ? "Loaded coverage diagnostics." : "Coverage preview failed.", payload.ok, renderCoveragePreview(payload));
          if (payload.ok) requestAnimationFrame(() => drawCoveragePreview(payload));
          log(`Coverage preview ${payload.ok ? "OK" : "FAILED"}: ${payload.path || path}`);
        })
        .catch((error) => {
          setPanel("coverage", "Coverage", String(error), false);
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
          `v${status.version} - python ${status.python_version} - ${status.python_ok ? "ready" : "python 3.10 expected"}`;
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
    document.getElementById("add-job").addEventListener("click", addJob);
    document.getElementById("save-batch-config").addEventListener("click", () => saveBatchConfig());
    document.getElementById("load-batch-config").addEventListener("click", loadBatchConfig);
    document.getElementById("batch-jobs-body").addEventListener("input", (event) => {
      const field = event.target.dataset.field;
      const index = Number(event.target.dataset.index);
      if (!field || Number.isNaN(index) || !batchJobs[index]) return;
      if (field === "args") {
        batchJobs[index].args = event.target.value.split(/\\s+/).filter(Boolean);
      } else {
        batchJobs[index][field] = event.target.value;
      }
    });
    document.getElementById("batch-jobs-body").addEventListener("click", (event) => {
      const removeIndex = event.target.dataset.remove;
      if (removeIndex === undefined) return;
      batchJobs.splice(Number(removeIndex), 1);
      renderBatchTable();
    });
    renderBatchTable();
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    });

    document.querySelectorAll(".workspace-tab[data-workspace]").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".workspace-tab[data-workspace]").forEach((t) => t.classList.toggle("active", t === tab));
        document.querySelectorAll(".workspace").forEach((workspace) => {
          workspace.hidden = workspace.id !== `workspace-${tab.dataset.workspace}`;
        });
      });
    });

    function postConsole(endpoint, body) {
      return fetch(endpoint, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body || {}),
      }).then((response) => response.json());
    }

    const CONSOLE_COLOR_MAPS = {
      heat: [[8, 12, 20], [21, 92, 106], [47, 217, 196], [242, 179, 61]],
      gray: [[0, 0, 0], [255, 255, 255]],
      hot: [[0, 0, 0], [160, 0, 0], [255, 160, 0], [255, 255, 200]],
      jet: [[0, 0, 143], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0], [128, 0, 0]],
      turbo: [[48, 18, 59], [70, 107, 227], [41, 187, 190], [126, 220, 80], [249, 206, 24], [233, 79, 30], [122, 4, 3]],
      viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]],
      inferno: [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
      magma: [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]],
      parula: [[53, 42, 135], [42, 73, 190], [32, 110, 196], [16, 145, 180], [62, 165, 143], [121, 178, 104], [177, 184, 73], [236, 187, 36], [249, 251, 14]],
      ocean: [[0, 0, 60], [0, 60, 120], [0, 140, 140], [100, 200, 180], [255, 255, 255]],
      rwb: [[33, 102, 172], [103, 169, 207], [209, 229, 240], [247, 247, 247], [253, 219, 199], [239, 138, 98], [178, 24, 43]],
      rkb: [[0, 60, 255], [0, 0, 0], [255, 60, 0]],
    };

    function createConsole(cfg) {
      const byId = (name) => document.getElementById(`${cfg.prefix}-${name}`);
      let shape = null;
      let lastPreview = null;
      let positionMode = null;
      let overlay = null;
      let pixelMm = 0.075;
      let zplotActive = false;
      let lastCurveXScale = null;
      let axisMode = "pixel";
      let fullBounds = null;
      let pixelSpacePreview = null;

      function setStatus(message, ok) {
        const node = byId("status");
        node.className = ok ? "muted" : "warn";
        node.textContent = message;
      }

      function setCurveStatus(message, ok) {
        const node = byId("curve-status");
        node.className = ok ? "muted" : "warn";
        node.textContent = message;
      }

      function applyCalibrationToFields(cal) {
        if (!cal) return;
        pixelMm = cal.pixel_mm || pixelMm;
        byId("energy").value = cal.energy_kev;
        byId("sdd").value = cal.sdd_mm;
        byId("beam-y").value = cal.beam_pos[0];
        byId("beam-z").value = cal.beam_pos[1];
        if (cfg.hasFootprint) byId("footprint").value = cal.footprint_mm;
        if (cfg.hasIncidence) {
          byId("ai-deg").value = cal.incident_angle_deg;
          incidenceFromDegree();
        }
      }

      function incidenceFromPixel() {
        const pix = Number(byId("ai-pix").value);
        const sdd = Number(byId("sdd").value);
        if (!Number.isFinite(pix) || !Number.isFinite(sdd) || sdd === 0) return;
        const ai = (Math.atan((pix * pixelMm) / sdd) * 180) / Math.PI / 2;
        byId("ai-deg").value = ai.toFixed(4);
      }

      function incidenceFromDegree() {
        const deg = Number(byId("ai-deg").value);
        const sdd = Number(byId("sdd").value);
        if (!Number.isFinite(deg) || !Number.isFinite(sdd) || pixelMm === 0) return;
        const pix = (sdd * Math.tan((deg * 2 * Math.PI) / 180)) / pixelMm;
        byId("ai-pix").value = pix.toFixed(2);
      }

      if (cfg.hasIncidence) {
        byId("ai-pix").addEventListener("change", incidenceFromPixel);
        byId("ai-deg").addEventListener("change", incidenceFromDegree);
        byId("ai-zplot").addEventListener("click", runZPlot);
      }

      function runZPlot() {
        if (!shape || !lastPreview) {
          setStatus("Load data first.", false);
          return;
        }
        const cols = shape[1];
        const rows = shape[0];
        const beamY = Number(byId("beam-y").value);
        const yOffset = Number(byId("ai-ypix").value) || 0;
        const targetCol = Math.min(cols, Math.max(1, Math.round(beamY + yOffset)));
        const previewRows = lastPreview.length;
        const previewCols = lastPreview[0].length;
        const colIndex = Math.min(previewCols - 1, Math.max(0, Math.round(((targetCol - 1) / Math.max(1, cols - 1)) * (previewCols - 1))));
        const rowPixels = [];
        const profile = [];
        for (let r = 0; r < previewRows; r += 1) {
          const value = lastPreview[r][colIndex];
          if (value === null || value === undefined || !Number.isFinite(value)) continue;
          const realRow = 1 + (r / Math.max(1, previewRows - 1)) * (rows - 1);
          rowPixels.push(realRow);
          profile.push(value);
        }
        if (!rowPixels.length) {
          setStatus("No finite data along that column for z-plot.", false);
          return;
        }
        drawCurve(rowPixels, profile, "pixel (row, z)");
        zplotActive = true;
        byId("curve-svg").style.cursor = "crosshair";
        const statusNode = byId("curve-status");
        if (statusNode) statusNode.textContent = "Click the peak on the profile below to set the incidence pixel.";
        log(`${cfg.mode.toUpperCase()} z-plot: sampling column pixel ${targetCol} (beam-y ${beamY} + y-pos ${yOffset}).`);
      }

      const LUT_STEPS = 256;
      const colorLutCache = {};

      function colorLut(mapName) {
        if (colorLutCache[mapName]) return colorLutCache[mapName];
        const stops = CONSOLE_COLOR_MAPS[mapName] || CONSOLE_COLOR_MAPS.heat;
        const lut = new Uint8Array(LUT_STEPS * 3);
        for (let i = 0; i < LUT_STEPS; i += 1) {
          const scaled = (i / (LUT_STEPS - 1)) * (stops.length - 1);
          const index = Math.min(stops.length - 2, Math.floor(scaled));
          const local = scaled - index;
          const a = stops[index];
          const b = stops[index + 1];
          lut[i * 3] = Math.round(a[0] + (b[0] - a[0]) * local);
          lut[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * local);
          lut[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * local);
        }
        colorLutCache[mapName] = lut;
        return lut;
      }

      function formatBoundValue(value) {
        if (!Number.isFinite(value)) return "0";
        return axisMode === "pixel" ? String(Math.round(value)) : value.toFixed(6);
      }

      function getViewWindow() {
        if (!shape || !fullBounds) return null;
        const yLo = Math.min(fullBounds.yStart, fullBounds.yEnd);
        const yHi = Math.max(fullBounds.yStart, fullBounds.yEnd);
        const zLo = Math.min(fullBounds.zStart, fullBounds.zEnd);
        const zHi = Math.max(fullBounds.zStart, fullBounds.zEnd);
        let yMin = Number(byId("yrange-min").value);
        let yMax = Number(byId("yrange-max").value);
        let zMin = Number(byId("zrange-min").value);
        let zMax = Number(byId("zrange-max").value);
        if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax <= yMin) { yMin = yLo; yMax = yHi; }
        if (!Number.isFinite(zMin) || !Number.isFinite(zMax) || zMax <= zMin) { zMin = zLo; zMax = zHi; }
        yMin = Math.max(yLo, Math.min(yMin, yHi));
        yMax = Math.max(yLo, Math.min(yMax, yHi));
        zMin = Math.max(zLo, Math.min(zMin, zHi));
        zMax = Math.max(zLo, Math.min(zMax, zHi));
        if (yMax <= yMin) { yMin = yLo; yMax = yHi; }
        if (zMax <= zMin) { zMin = zLo; zMax = zHi; }
        return {yMin, yMax, zMin, zMax, rows: shape[0], cols: shape[1]};
      }

      function valueToAxisFraction(value, start, end) {
        const span = end - start;
        if (Math.abs(span) < 1e-12) return 0;
        return (value - start) / span;
      }

      function cropToViewWindow(preview, view) {
        if (!view || !fullBounds) return preview;
        const previewRows = preview.length;
        const previewCols = preview[0].length;
        const fracColA = valueToAxisFraction(view.yMin, fullBounds.yStart, fullBounds.yEnd);
        const fracColB = valueToAxisFraction(view.yMax, fullBounds.yStart, fullBounds.yEnd);
        const fracRowA = valueToAxisFraction(view.zMin, fullBounds.zStart, fullBounds.zEnd);
        const fracRowB = valueToAxisFraction(view.zMax, fullBounds.zStart, fullBounds.zEnd);
        const colFracLo = Math.min(fracColA, fracColB);
        const colFracHi = Math.max(fracColA, fracColB);
        const rowFracLo = Math.min(fracRowA, fracRowB);
        const rowFracHi = Math.max(fracRowA, fracRowB);
        const colStart = Math.max(0, Math.min(previewCols - 1, Math.round(colFracLo * (previewCols - 1))));
        const colEnd = Math.max(colStart, Math.min(previewCols - 1, Math.round(colFracHi * (previewCols - 1))));
        const rowStart = Math.max(0, Math.min(previewRows - 1, Math.round(rowFracLo * (previewRows - 1))));
        const rowEnd = Math.max(rowStart, Math.min(previewRows - 1, Math.round(rowFracHi * (previewRows - 1))));
        if (rowStart === 0 && rowEnd === previewRows - 1 && colStart === 0 && colEnd === previewCols - 1) return preview;
        const cropped = [];
        for (let r = rowStart; r <= rowEnd; r += 1) {
          cropped.push(preview[r].slice(colStart, colEnd + 1));
        }
        return cropped;
      }

      function renderCanvas(values) {
        if (values) lastPreview = values;
        const fullPreview = lastPreview;
        const canvas = byId("canvas");
        if (!fullPreview || !fullPreview.length || !fullPreview[0].length) return;
        const view = getViewWindow();
        const preview = cropToViewWindow(fullPreview, view);
        if (!preview.length || !preview[0].length) return;
        const rows = preview.length;
        const cols = preview[0].length;
        canvas.width = cols;
        canvas.height = rows;
        canvas.style.aspectRatio = `${cols} / ${rows}`;
        const ctx = canvas.getContext("2d");
        const image = ctx.createImageData(cols, rows);

        const scaleMode = byId("color-scale").value;
        const lut = colorLut(byId("colormap").value);
        let cmin = Number(byId("cmin").value);
        let cmax = Number(byId("cmax").value);
        if (!Number.isFinite(cmin) || !Number.isFinite(cmax) || cmax <= cmin) {
          let finiteMin = Infinity;
          let finiteMax = -Infinity;
          for (let ry = 0; ry < rows; ry += 1) {
            const row = preview[ry];
            for (let rx = 0; rx < cols; rx += 1) {
              const v = row[rx];
              if (v !== null && v !== undefined && Number.isFinite(v)) {
                if (v < finiteMin) finiteMin = v;
                if (v > finiteMax) finiteMax = v;
              }
            }
          }
          cmin = Number.isFinite(finiteMin) ? finiteMin : 0;
          cmax = Number.isFinite(finiteMax) ? finiteMax : 1;
        }
        const isLog = scaleMode === "Log";
        const logLo = cmin > 0 ? Math.log10(cmin) : 0;
        const logHi = cmax > 0 ? Math.log10(cmax) : 1;
        const logSpan = Math.max(logHi - logLo, 1e-9);
        const linSpan = Math.max(cmax - cmin, 1e-9);
        const data = image.data;

        for (let y = 0; y < rows; y += 1) {
          const row = preview[y];
          const rowOffset = y * cols * 4;
          for (let x = 0; x < cols; x += 1) {
            const offset = rowOffset + x * 4;
            const value = row[x];
            if (value === null || value === undefined || !Number.isFinite(value)) {
              data[offset] = 255;
              data[offset + 1] = 255;
              data[offset + 2] = 255;
              data[offset + 3] = 255;
              continue;
            }
            const t = isLog
              ? ((value > 0 ? Math.log10(value) : logLo) - logLo) / logSpan
              : (value - cmin) / linSpan;
            const lutIndex = Math.round(Math.max(0, Math.min(1, t)) * (LUT_STEPS - 1)) * 3;
            data[offset] = lut[lutIndex];
            data[offset + 1] = lut[lutIndex + 1];
            data[offset + 2] = lut[lutIndex + 2];
            data[offset + 3] = 255;
          }
        }
        ctx.putImageData(image, 0, 0);
        renderColorbar(lut, cmin, cmax);
      }

      function formatCbarValue(value) {
        if (!Number.isFinite(value)) return "n/a";
        const abs = Math.abs(value);
        if (abs !== 0 && (abs < 0.01 || abs >= 100000)) return value.toExponential(2);
        return Number(value.toPrecision(4)).toString();
      }

      function renderColorbar(lut, cmin, cmax) {
        const canvas = byId("colorbar");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const width = canvas.width;
        const height = canvas.height;
        const image = ctx.createImageData(width, height);
        for (let y = 0; y < height; y += 1) {
          const t = 1 - y / Math.max(1, height - 1);
          const lutIndex = Math.round(t * (LUT_STEPS - 1)) * 3;
          for (let x = 0; x < width; x += 1) {
            const offset = (y * width + x) * 4;
            image.data[offset] = lut[lutIndex];
            image.data[offset + 1] = lut[lutIndex + 1];
            image.data[offset + 2] = lut[lutIndex + 2];
            image.data[offset + 3] = 255;
          }
        }
        ctx.putImageData(image, 0, 0);
        byId("cbar-min").textContent = formatCbarValue(cmin);
        byId("cbar-max").textContent = formatCbarValue(cmax);
      }

      function drawOverlay() {
        if (!overlay || !shape || axisMode !== "pixel") return;
        const canvas = byId("canvas");
        const ctx = canvas.getContext("2d");
        const beamY = Number(byId("beam-y").value);
        const beamZ = Number(byId("beam-z").value);
        const view = getViewWindow() || {yMin: 1, yMax: shape[1], zMin: 1, zMax: shape[0]};
        const cx = ((beamY - view.yMin) / Math.max(1, view.yMax - view.yMin)) * canvas.width;
        const cy = ((beamZ - view.zMin) / Math.max(1, view.zMax - view.zMin)) * canvas.height;
        ctx.save();
        ctx.strokeStyle = "rgba(26, 77, 128, 0.9)";
        ctx.lineWidth = Math.max(1, canvas.width / 200);
        if (overlay.type === "line") {
          const rad = (overlay.angleDeg * Math.PI) / 180;
          const dx = Math.cos(rad);
          const dy = -Math.sin(rad);
          const length = Math.max(canvas.width, canvas.height);
          ctx.beginPath();
          ctx.moveTo(cx - dx * length, cy - dy * length);
          ctx.lineTo(cx + dx * length, cy + dy * length);
          ctx.stroke();
        } else if (overlay.type === "azimuthal") {
          const scale = canvas.width / Math.max(1, view.yMax - view.yMin);
          ctx.beginPath();
          ctx.arc(cx, cy, overlay.radius * scale, 0, 2 * Math.PI);
          ctx.stroke();
        }
        ctx.restore();
      }

      function renderAxisTicks() {
        const view = getViewWindow();
        const yEl = byId("axis-y");
        const xEl = byId("axis-x");
        if (!view) {
          if (yEl) yEl.querySelectorAll(".axis-tick").forEach((el) => { el.textContent = ""; });
          if (xEl) xEl.querySelectorAll(".axis-tick").forEach((el) => { el.textContent = ""; });
          return;
        }
        const formatTick = axisMode === "pixel" ? (v) => String(Math.round(v)) : formatCbarValue;
        const zVals = [formatTick(view.zMin), formatTick((view.zMin + view.zMax) / 2), formatTick(view.zMax)];
        const yVals = [formatTick(view.yMin), formatTick((view.yMin + view.yMax) / 2), formatTick(view.yMax)];
        if (yEl) {
          const ticks = yEl.querySelectorAll(".axis-tick");
          ticks.forEach((el, i) => { el.textContent = zVals[i] ?? ""; });
        }
        if (xEl) {
          const ticks = xEl.querySelectorAll(".axis-tick");
          ticks.forEach((el, i) => { el.textContent = yVals[i] ?? ""; });
        }
        const fontSz = Number(byId("font-sz").value);
        const fontPx = Number.isFinite(fontSz) && fontSz > 0 ? `${Math.max(8, Math.min(22, fontSz))}px` : "";
        if (fontPx) {
          if (yEl) yEl.style.fontSize = fontPx;
          if (xEl) xEl.style.fontSize = fontPx;
        }
      }

      function redraw() {
        renderCanvas(null);
        drawOverlay();
        renderAxisTicks();
      }

      function clickToPixel(event) {
        if (!shape || axisMode !== "pixel") return null;
        const canvas = byId("canvas");
        const rect = canvas.getBoundingClientRect();
        const xFrac = (event.clientX - rect.left) / rect.width;
        const yFrac = (event.clientY - rect.top) / rect.height;
        const view = getViewWindow() || {yMin: 1, yMax: shape[1], zMin: 1, zMax: shape[0]};
        return {
          x: Math.min(view.yMax, Math.max(view.yMin, Math.round(view.yMin + xFrac * (view.yMax - view.yMin)))),
          y: Math.min(view.zMax, Math.max(view.zMin, Math.round(view.zMin + yFrac * (view.zMax - view.zMin)))),
        };
      }

      byId("canvas").addEventListener("click", (event) => {
        const pixel = clickToPixel(event);
        if (!pixel || !positionMode) return;
        const beamY = Number(byId("beam-y").value);
        const beamZ = Number(byId("beam-z").value);
        if (positionMode === "line") {
          const angle = ((Math.atan2(-(pixel.y - beamZ), -(pixel.x - beamY)) * 180) / Math.PI + 360) % 360;
          byId("line-angle").value = angle.toFixed(2);
          log(`${cfg.mode.toUpperCase()} line position clicked at pixel (${pixel.x}, ${pixel.y}) -> angle ${angle.toFixed(2)} deg.`);
        } else if (positionMode === "azimuthal") {
          const radius = Math.hypot(pixel.x - beamY, pixel.y - beamZ);
          byId("azi-pixel").value = radius.toFixed(1);
          log(`${cfg.mode.toUpperCase()} azimuthal position clicked at pixel (${pixel.x}, ${pixel.y}) -> radius ${radius.toFixed(1)} px.`);
        }
      });

      function applyZoom(factor, xFrac, yFrac) {
        if (!shape || !fullBounds) return;
        const yLo = Math.min(fullBounds.yStart, fullBounds.yEnd);
        const yHi = Math.max(fullBounds.yStart, fullBounds.yEnd);
        const zLo = Math.min(fullBounds.zStart, fullBounds.zEnd);
        const zHi = Math.max(fullBounds.zStart, fullBounds.zEnd);
        const view = getViewWindow() || {yMin: yLo, yMax: yHi, zMin: zLo, zMax: zHi};
        const yCenter = view.yMin + xFrac * (view.yMax - view.yMin);
        const zCenter = view.zMin + yFrac * (view.zMax - view.zMin);
        const minYSpan = (yHi - yLo) * 0.002;
        const minZSpan = (zHi - zLo) * 0.002;
        const ySpan = Math.max(minYSpan, (view.yMax - view.yMin) * factor);
        const zSpan = Math.max(minZSpan, (view.zMax - view.zMin) * factor);
        let yMin = yCenter - ySpan / 2;
        let yMax = yCenter + ySpan / 2;
        let zMin = zCenter - zSpan / 2;
        let zMax = zCenter + zSpan / 2;
        if (yMin < yLo) { yMax += yLo - yMin; yMin = yLo; }
        if (yMax > yHi) { yMin -= yMax - yHi; yMax = yHi; }
        if (zMin < zLo) { zMax += zLo - zMin; zMin = zLo; }
        if (zMax > zHi) { zMin -= zMax - zHi; zMax = zHi; }
        byId("yrange-min").value = formatBoundValue(Math.max(yLo, yMin));
        byId("yrange-max").value = formatBoundValue(Math.min(yHi, yMax));
        byId("zrange-min").value = formatBoundValue(Math.max(zLo, zMin));
        byId("zrange-max").value = formatBoundValue(Math.min(zHi, zMax));
        redraw();
      }

      function resetZoom() {
        if (!shape || !fullBounds) return;
        byId("yrange-min").value = formatBoundValue(Math.min(fullBounds.yStart, fullBounds.yEnd));
        byId("yrange-max").value = formatBoundValue(Math.max(fullBounds.yStart, fullBounds.yEnd));
        byId("zrange-min").value = formatBoundValue(Math.min(fullBounds.zStart, fullBounds.zEnd));
        byId("zrange-max").value = formatBoundValue(Math.max(fullBounds.zStart, fullBounds.zEnd));
        redraw();
      }

      byId("canvas").addEventListener("wheel", (event) => {
        if (!shape) return;
        event.preventDefault();
        const canvas = byId("canvas");
        const rect = canvas.getBoundingClientRect();
        const xFrac = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
        const yFrac = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
        const factor = event.deltaY < 0 ? 0.8 : 1.25;
        applyZoom(factor, xFrac, yFrac);
      }, {passive: false});

      byId("canvas").addEventListener("dblclick", () => resetZoom());
      byId("reset-zoom").addEventListener("click", () => resetZoom());

      function drawCurve(xValues, yValues, xLabel) {
        const svg = byId("curve-svg");
        svg.innerHTML = "";
        const width = 640;
        const height = 260;
        const pad = 34;
        const points = xValues
          .map((x, index) => [x, yValues[index]])
          .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
        if (!points.length) return;
        const xs = points.map((p) => p[0]);
        const ys = points.map((p) => p[1]);
        const xMin = Math.min(...xs);
        const xMax = Math.max(...xs);
        const yMin = Math.min(...ys);
        const yMax = Math.max(...ys);
        const xScale = (v) => pad + ((v - xMin) / Math.max(xMax - xMin, 1e-12)) * (width - pad * 2);
        const yScale = (v) => height - pad - ((v - yMin) / Math.max(yMax - yMin, 1e-12)) * (height - pad * 2);
        const path = points
          .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${xScale(x).toFixed(2)} ${yScale(y).toFixed(2)}`)
          .join(" ");
        svg.insertAdjacentHTML("beforeend", `<path class="curve-path" d="${path}"></path>`);
        svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#adadad"></line>`);
        svg.insertAdjacentHTML("beforeend", `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#adadad"></line>`);
        svg.insertAdjacentHTML("beforeend", `<text x="${pad}" y="22" fill="#5c5c5c" font-size="12">intensity</text>`);
        svg.insertAdjacentHTML("beforeend", `<text x="${width - pad - 50}" y="${height - 8}" fill="#5c5c5c" font-size="12">${escapeHtml(xLabel)}</text>`);
        lastCurveXScale = {xMin, xMax, pad, width};
      }

      if (cfg.hasIncidence) {
        byId("curve-svg").addEventListener("click", (event) => {
          if (!zplotActive || !lastCurveXScale) return;
          const svg = byId("curve-svg");
          const rect = svg.getBoundingClientRect();
          const svgX = ((event.clientX - rect.left) / rect.width) * lastCurveXScale.width;
          const frac = (svgX - lastCurveXScale.pad) / (lastCurveXScale.width - lastCurveXScale.pad * 2);
          const clamped = Math.min(1, Math.max(0, frac));
          const pixelVal = lastCurveXScale.xMin + clamped * (lastCurveXScale.xMax - lastCurveXScale.xMin);
          byId("ai-pix").value = pixelVal.toFixed(2);
          incidenceFromPixel();
          zplotActive = false;
          byId("curve-svg").style.cursor = "default";
          const statusNode = byId("curve-status");
          if (statusNode) statusNode.textContent = `Incidence pixel set to ${pixelVal.toFixed(2)} from z-plot (ai = ${byId("ai-deg").value} deg).`;
          log(`${cfg.mode.toUpperCase()} z-plot picked pixel ${pixelVal.toFixed(2)} -> ai ${byId("ai-deg").value} deg.`);
        });
      }

      function browseDirectory(dirId, fileSelectId, statusFn, suffixes) {
        const directory = byId(dirId).value.trim();
        if (!directory) {
          statusFn("Directory is required.", false);
          return;
        }
        const suffixParam = suffixes ? `&suffixes=${encodeURIComponent(suffixes)}` : "";
        fetch(`/api/console/list-files?directory=${encodeURIComponent(directory)}${suffixParam}`)
          .then((response) => response.json())
          .then((payload) => {
            const select = byId(fileSelectId);
            select.innerHTML = "";
            if (!payload.ok) {
              select.appendChild(new Option(payload.message || "Browse failed.", ""));
              statusFn(payload.message || "Browse failed.", false);
              log(`Browse FAILED: ${payload.message}`);
              return;
            }
            if (!payload.files.length) {
              select.appendChild(new Option("(no files found)", ""));
              statusFn(`No files found in ${directory}.`, false);
              return;
            }
            payload.files.forEach((name) => select.appendChild(new Option(name, name)));
            statusFn(`Found ${payload.files.length} file(s) in ${directory}.`, true);
            log(`Browse OK: ${directory} (${payload.files.length} files)`);
          })
          .catch((error) => {
            statusFn(String(error), false);
            log(String(error));
          });
      }

      byId("browse").addEventListener("click", () => browseDirectory("dir", "file", setStatus));
      byId("bkg-browse").addEventListener("click", () => browseDirectory("bkg-dir", "bkg-file", setStatus));
      byId("info-browse").addEventListener("click", () => browseDirectory("info-dir", "info-file", setStatus, "json"));

      function pollLoadProgress() {
        const interval = setInterval(() => {
          fetch("/api/console/load-progress")
            .then((response) => response.json())
            .then((progress) => {
              if (progress.active) {
                setStatus(`Loading file ${progress.current}/${progress.total}: ${progress.label}...`, true);
              }
            })
            .catch(() => {});
        }, 200);
        return () => clearInterval(interval);
      }

      byId("load").addEventListener("click", () => {
        const directory = byId("dir").value.trim();
        const filename = byId("file").value;
        if (!directory || !filename) {
          setStatus("Browse a directory and pick a file first.", false);
          return;
        }
        const multishot = byId("multishot").checked;
        log(`${cfg.mode.toUpperCase()}: loading ${filename} from ${directory}${multishot ? " (multishot)" : ""}`);
        if (multishot) setStatus("Starting multishot load...", true);
        const stopPolling = multishot ? pollLoadProgress() : null;
        postConsole("/api/console/load", {
          directory,
          filename,
          mode: cfg.mode,
          detector: byId("detector").value,
          multishot,
        })
          .then((payload) => {
            if (stopPolling) stopPolling();
            if (!payload.ok) {
              setStatus(payload.message || "Load failed.", false);
              log(`Load FAILED: ${payload.message}`);
              return;
            }
            shape = payload.shape;
            overlay = null;
            axisMode = "pixel";
            fullBounds = {yStart: 1, yEnd: shape[1], zStart: 1, zEnd: shape[0]};
            pixelSpacePreview = payload.preview;
            if (byId("map-format")) byId("map-format").value = "Pixel";
            byId("ic").textContent = payload.ic ?? "0";
            byId("bc").textContent = payload.bc ?? "0";
            applyCalibrationToFields(payload.calibration);
            byId("qmap-cal").classList.remove("ok");
            byId("grid-remove").classList.remove("ok");
            byId("q-range").textContent = "Q range: not computed yet.";
            byId("yrange-min").value = 1;
            byId("yrange-max").value = shape[1];
            byId("zrange-min").value = 1;
            byId("zrange-max").value = shape[0];
            renderCanvas(payload.preview);
            renderAxisTicks();
            const seriesNote = payload.series_file_count ? ` (averaged ${payload.series_file_count} files)` : "";
            setStatus(`Loaded ${payload.shape.join(" x ")} from ${payload.path}.${seriesNote}`, true);
            log(`Load OK: ${payload.path} (${payload.shape.join("x")})${seriesNote}`);
          })
          .catch((error) => {
            if (stopPolling) stopPolling();
            setStatus(String(error), false);
            log(String(error));
          });
      });

      byId("load-info").addEventListener("click", () => {
        const infoDir = byId("info-dir").value.trim();
        const infoFile = byId("info-file").value;
        if (!infoDir || !infoFile) {
          setStatus("Browse a directory and pick a calibration file first.", false);
          return;
        }
        postConsole("/api/console/load-info", {directory: infoDir, filename: infoFile}).then((payload) => {
          if (!payload.ok) {
            setStatus(payload.message || "Load Info failed.", false);
            log(`Load Info FAILED: ${payload.message}`);
            return;
          }
          applyCalibrationToFields(payload.calibration);
          setStatus("Calibration loaded.", true);
          log(`Load Info OK: ${payload.path}`);
        });
      });

      byId("grid-remove").addEventListener("click", () => {
        postConsole("/api/console/grid-remove", {}).then((payload) => {
          if (!payload.ok) {
            setStatus(payload.message || "Grid remove failed.", false);
            log(`Grid Remove FAILED: ${payload.message}`);
            return;
          }
          byId("grid-remove").classList.add("ok");
          axisMode = "pixel";
          pixelSpacePreview = payload.preview;
          if (byId("map-format")) byId("map-format").value = "Pixel";
          renderCanvas(payload.preview);
          renderAxisTicks();
          setStatus(payload.message, true);
          log(`Grid Remove OK: ${payload.message}`);
        });
      });

      byId("load-bkg").addEventListener("click", () => {
        const directory = byId("bkg-dir").value.trim();
        const filename = byId("bkg-file").value;
        if (!directory || !filename) {
          setStatus("Browse a background directory and pick a file first.", false);
          return;
        }
        const multishot = byId("multishot").checked;
        if (multishot) setStatus("Starting multishot background load...", true);
        const stopPolling = multishot ? pollLoadProgress() : null;
        postConsole("/api/console/background", {directory, filename, multishot}).then((payload) => {
          if (stopPolling) stopPolling();
          if (!payload.ok) {
            setStatus(payload.message || "Background load failed.", false);
            log(`Load Bkg. FAILED: ${payload.message}`);
            return;
          }
          setStatus(`Background loaded from ${payload.path}.`, true);
          log(`Load Bkg. OK: ${payload.path}`);
        });
      });

      byId("subtract").addEventListener("click", () => {
        const method = byId("normal-method").value;
        const plusData = Number(byId("plus-data").value) || 1;
        postConsole("/api/console/normalize", {method, plus_data: plusData, confirm: false}).then((preview) => {
          if (!preview.ok) {
            setStatus(preview.message || "Normalization failed.", false);
            log(`BKG sub. & Norm FAILED: ${preview.message}`);
            return;
          }
          renderCanvas(preview.preview);
          const keep = window.confirm("Preview updated. Keep this background-subtracted/normalized result?");
          if (!keep) {
            renderCanvas(lastPreview);
            setStatus("Normalization discarded.", true);
            log("BKG sub. & Norm: discarded by user.");
            return;
          }
          postConsole("/api/console/normalize", {method, plus_data: plusData, confirm: true}).then((committed) => {
            if (!committed.ok) {
              setStatus(committed.message || "Normalization failed.", false);
              log(`BKG sub. & Norm FAILED: ${committed.message}`);
              return;
            }
            axisMode = "pixel";
            pixelSpacePreview = committed.preview;
            if (byId("map-format")) byId("map-format").value = "Pixel";
            renderCanvas(committed.preview);
            renderAxisTicks();
            setStatus(committed.message, true);
            log(`BKG sub. & Norm OK: ${committed.message}`);
          });
        });
      });

      byId("qmap-cal").addEventListener("click", () => {
        const payload = {
          energy_kev: Number(byId("energy").value),
          sdd_mm: Number(byId("sdd").value),
          beam_y: Number(byId("beam-y").value),
          beam_z: Number(byId("beam-z").value),
          incident_angle_deg: cfg.hasIncidence ? Number(byId("ai-deg").value) : 0,
          footprint_mm: cfg.hasFootprint ? Number(byId("footprint").value) : 0,
        };
        postConsole("/api/console/qmap", payload).then((result) => {
          if (!result.ok) {
            setStatus(result.message || "q-Map Cal. failed.", false);
            log(`q-Map Cal. FAILED: ${result.message}`);
            return;
          }
          byId("qmap-cal").classList.add("ok");
          const q = result.q_range || {};
          byId("q-range").textContent =
            `Q range: ${q.min !== null && q.min !== undefined ? q.min.toFixed(3) : "n/a"} to ${q.max !== null && q.max !== undefined ? q.max.toFixed(3) : "n/a"} 1/A`;
          axisMode = "pixel";
          pixelSpacePreview = result.preview;
          if (byId("map-format")) byId("map-format").value = "Pixel";
          renderCanvas(result.preview);
          renderAxisTicks();
          setStatus("q-Map Cal. complete.", true);
          log("q-Map Cal. OK.");
        });
      });

      if (byId("line-position")) {
        byId("line-position").addEventListener("click", () => {
          positionMode = "line";
          setStatus("Click a point on the 2D image to set the line angle.", true);
        });
      }

      byId("line-draw").addEventListener("click", () => {
        const angleDeg = Number(byId("line-angle").value) || 0;
        overlay = {type: "line", angleDeg};
        redraw();
        log(`Line overlay drawn at angle ${angleDeg} deg.`);
      });

      byId("line-plot").addEventListener("click", () => {
        const payload = {
          angle_deg: Number(byId("line-angle").value) || 0,
          cakecut: cfg.hasCakecut ? byId("cakecut").checked : false,
          range_minus: cfg.hasCakecut ? Number(byId("cakecut-minus").value) : -1,
          range_plus: cfg.hasCakecut ? Number(byId("cakecut-plus").value) : 1,
          use_median: byId("line-median").checked,
        };
        postConsole("/api/console/1d/line", payload).then((result) => {
          if (!result.ok) {
            setCurveStatus(result.message || "1D line plot failed.", false);
            log(`1d Plot FAILED: ${result.message}`);
            return;
          }
          drawCurve(result.preview.q, result.preview.intensity, "q [1/A]");
          const diag = result.coverage_diagnostics || {};
          setCurveStatus(`Line 1D ready. Coverage median ${diag.coverage_median ?? "n/a"}, low bins ${diag.low_bin_count ?? 0}, drops ${diag.drop_bin_count ?? 0}.`, true);
          log("1d Plot OK.");
        });
      });

      byId("azi-position").addEventListener("click", () => {
        positionMode = "azimuthal";
        setStatus("Click a point on the 2D image to set the azimuthal radius.", true);
      });

      byId("azi-draw").addEventListener("click", () => {
        const pixel = Number(byId("azi-pixel").value) || 0;
        overlay = {type: "azimuthal", radius: pixel};
        redraw();
        log(`Azimuthal ring overlay drawn at pixel radius ${pixel}.`);
      });

      byId("azi-plot").addEventListener("click", () => {
        const payload = {
          pixel: Number(byId("azi-pixel").value) || 0,
          range_minus: Number(byId("azi-range-minus").value),
          range_plus: Number(byId("azi-range-plus").value),
          mean_or_median: byId("azi-mean-median").value,
        };
        postConsole("/api/console/1d/azimuthal", payload).then((result) => {
          if (!result.ok) {
            setCurveStatus(result.message || "Azimuthal plot failed.", false);
            log(`Azimuthal Plot FAILED: ${result.message}`);
            return;
          }
          drawCurve(result.preview.angle, result.preview.intensity, "angle [deg]");
          const diag = result.coverage_diagnostics || {};
          setCurveStatus(`Azimuthal profile ready. Coverage median ${diag.coverage_median ?? "n/a"}, low bins ${diag.low_bin_count ?? 0}, drops ${diag.drop_bin_count ?? 0}.`, true);
          log("Azimuthal Plot OK.");
        });
      });

      if (cfg.hasCircular && byId("circular-plot")) {
        byId("circular-plot").addEventListener("click", () => {
          const payload = {
            angle_min: Number(byId("ang-min").value) || 0,
            angle_max: Number(byId("ang-max").value) || 360,
            min_coverage: Number(byId("min-coverage").value),
            use_median: byId("median").checked,
            calc_std: byId("std").checked,
          };
          postConsole("/api/console/1d/circular", payload).then((result) => {
            if (!result.ok) {
              setCurveStatus(result.message || "Circular average failed.", false);
              log(`Circular Average FAILED: ${result.message}`);
              return;
            }
            drawCurve(result.preview.q, result.preview.intensity, "q [1/A]");
            const diag = result.coverage_diagnostics || {};
            setCurveStatus(`Circular average ready. Coverage median ${diag.coverage_median ?? "n/a"}, low bins ${diag.low_bin_count ?? 0}, drops ${diag.drop_bin_count ?? 0}.`, true);
            log("Circular Average OK.");
          });
        });
      }

      ["color-scale", "colormap", "cmin", "cmax", "yrange-min", "yrange-max", "zrange-min", "zrange-max", "font-sz"].forEach((name) => {
        byId(name).addEventListener("change", redraw);
      });
      byId("update").addEventListener("click", redraw);

      byId("map-format").addEventListener("change", () => {
        const mode = byId("map-format").value;
        if (mode === "Pixel") {
          if (!pixelSpacePreview || !shape) {
            setStatus("Load data first.", false);
            return;
          }
          axisMode = "pixel";
          fullBounds = {yStart: 1, yEnd: shape[1], zStart: 1, zEnd: shape[0]};
          byId("yrange-min").value = 1;
          byId("yrange-max").value = shape[1];
          byId("zrange-min").value = 1;
          byId("zrange-max").value = shape[0];
          renderCanvas(pixelSpacePreview);
          renderAxisTicks();
          setStatus("Pixel view restored.", true);
          log(`${cfg.mode.toUpperCase()} map format -> Pixel.`);
          return;
        }
        if (!shape) {
          setStatus("Load data first.", false);
          byId("map-format").value = "Pixel";
          return;
        }
        setStatus(`Computing ${mode} image...`, true);
        postConsole("/api/console/qspace-image", {mapping: mode}).then((payload) => {
          if (!payload.ok) {
            setStatus(payload.message || `${mode} transform failed.`, false);
            log(`${cfg.mode.toUpperCase()} qspace-image FAILED: ${payload.message}`);
            byId("map-format").value = "Pixel";
            return;
          }
          axisMode = "q";
          fullBounds = {
            yStart: payload.horizontal_range.min,
            yEnd: payload.horizontal_range.max,
            zStart: payload.vertical_range.max,
            zEnd: payload.vertical_range.min,
          };
          byId("yrange-min").value = formatBoundValue(payload.horizontal_range.min);
          byId("yrange-max").value = formatBoundValue(payload.horizontal_range.max);
          byId("zrange-min").value = formatBoundValue(payload.vertical_range.min);
          byId("zrange-max").value = formatBoundValue(payload.vertical_range.max);
          renderCanvas(payload.preview);
          renderAxisTicks();
          setStatus(`${mode} image ready.`, true);
          log(`${cfg.mode.toUpperCase()} qspace-image OK (${mode}).`);
        });
      });

      byId("save").addEventListener("click", () => {
        postConsole("/api/console/save", {}).then((result) => {
          if (!result.ok) {
            setStatus(result.message || "Save failed.", false);
            log(`Save Img FAILED: ${result.message}`);
            return;
          }
          setStatus(`Saved ${result.h5_path}${result.curve_path ? " and " + result.curve_path : ""}.`, true);
          log(`Save Img OK: ${result.h5_path}`);
          const format = byId("save-format").value === "jpg" ? "jpg" : "png";
          const mime = format === "jpg" ? "image/jpeg" : "image/png";
          const canvas = byId("canvas");
          const link = document.createElement("a");
          link.href = canvas.toDataURL(mime, 0.92);
          link.download = `pgixs_console_${cfg.mode}_2d.${format}`;
          link.click();
        });
      });
    }

    createConsole({mode: "gixs", prefix: "c", hasIncidence: true, hasFootprint: true, hasCakecut: true, hasCircular: false});
    createConsole({mode: "trxs", prefix: "t", hasIncidence: false, hasFootprint: false, hasCakecut: false, hasCircular: true});
  </script>
</body>
</html>
"""

INDEX_HTML = INDEX_HTML.replace("__CONSOLE_GIXS_HTML__", _CONSOLE_GIXS_HTML)
INDEX_HTML = INDEX_HTML.replace("__CONSOLE_TRXS_HTML__", _CONSOLE_TRXS_HTML)


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
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._send_json(status_payload())
            return
        if path == "/api/workflows":
            self._send_json(workflows_payload())
            return
        if path == "/api/startup":
            self._send_json(startup_payload())
            return
        if path == "/api/batch/load":
            query = parse_qs(parsed.query)
            file_path = (query.get("path") or [""])[0]
            self._send_json(batch_load_payload(file_path))
            return
        if path == "/api/console/state":
            self._send_json(console_state())
            return
        if path == "/api/console/load-progress":
            self._send_json(console_load_progress())
            return
        if path == "/api/console/list-files":
            query = parse_qs(parsed.query)
            directory = (query.get("directory") or [""])[0]
            suffixes_raw = (query.get("suffixes") or [""])[0]
            suffixes = [item for item in suffixes_raw.split(",") if item] if suffixes_raw else None
            self._send_json(console_list_files({"directory": directory, "suffixes": suffixes}))
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
        if self.path == "/api/batch/save":
            self._send_json(batch_save_payload(self._read_json()))
            return
        if self.path == "/api/console/load":
            self._send_json(console_load(self._read_json()))
            return
        if self.path == "/api/console/load-info":
            self._send_json(console_load_info(self._read_json()))
            return
        if self.path == "/api/console/grid-remove":
            self._send_json(console_grid_remove(self._read_json()))
            return
        if self.path == "/api/console/background":
            self._send_json(console_background_load(self._read_json()))
            return
        if self.path == "/api/console/normalize":
            self._send_json(console_normalize(self._read_json()))
            return
        if self.path == "/api/console/qmap":
            self._send_json(console_qmap(self._read_json()))
            return
        if self.path == "/api/console/qspace-image":
            self._send_json(console_qspace_image(self._read_json()))
            return
        if self.path == "/api/console/1d/line":
            self._send_json(console_line_1d(self._read_json()))
            return
        if self.path == "/api/console/1d/azimuthal":
            self._send_json(console_azimuthal_1d(self._read_json()))
            return
        if self.path == "/api/console/1d/circular":
            self._send_json(console_circular_average(self._read_json()))
            return
        if self.path == "/api/console/save":
            self._send_json(console_save(self._read_json()))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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


def batch_load_payload(file_path: str) -> dict[str, Any]:
    path_str = str(file_path or "").strip()
    if not path_str:
        return {"ok": False, "path": path_str, "message": "path is required."}
    try:
        jobs = load_batch_jobs(path_str)
    except Exception as exc:  # noqa: BLE001 - UI endpoint should return visible failure details.
        return {"ok": False, "path": path_str, "message": str(exc)}
    return {
        "ok": True,
        "path": path_str,
        "jobs": [asdict(job) for job in jobs],
    }


def batch_save_payload(payload: dict[str, Any]) -> dict[str, Any]:
    path_str = str(payload.get("path", "")).strip()
    raw_jobs = payload.get("jobs", [])
    if not path_str:
        return {"ok": False, "path": path_str, "message": "path is required."}
    if not isinstance(raw_jobs, list) or not all(isinstance(item, dict) for item in raw_jobs):
        return {"ok": False, "path": path_str, "message": "jobs must be a list of objects."}
    try:
        target = Path(path_str)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"jobs": raw_jobs}, indent=2, ensure_ascii=False), encoding="utf-8")
        jobs = load_batch_jobs(target)
    except Exception as exc:  # noqa: BLE001 - UI endpoint should return visible failure details.
        return {"ok": False, "path": path_str, "message": str(exc)}
    return {"ok": True, "path": path_str, "job_count": len(jobs)}


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
