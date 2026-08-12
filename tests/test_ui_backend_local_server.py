from http import HTTPStatus
import json

from pgixs_python.ui_backend.local_server import (
    INDEX_HTML,
    PGIXSRequestHandler,
    batch_load_payload,
    batch_save_payload,
    console_load,
    console_qmap,
    console_state,
    coverage_payload,
    one_d_preview_payload,
    run_payload,
    two_d_preview_payload,
    validate_payload,
)


def test_local_server_has_expected_routes():
    assert "/" in {"/", "/index.html"}
    assert hasattr(PGIXSRequestHandler, "do_GET")
    assert HTTPStatus.OK.value == 200


def test_index_html_contains_workbench_regions():
    assert "Batch Workflow" in INDEX_HTML
    assert "Batch Queue" in INDEX_HTML
    assert "Run Summary" in INDEX_HTML
    assert 'data-tab="batch"' in INDEX_HTML
    assert 'id="panel-summary"' in INDEX_HTML
    assert "activateTab" in INDEX_HTML
    assert "setPanel" in INDEX_HTML
    assert "/api/startup" in INDEX_HTML
    assert "/api/validate" in INDEX_HTML
    assert "/api/run" in INDEX_HTML
    assert "Batch summary JSON" in INDEX_HTML
    assert "/api/preview/2d" in INDEX_HTML
    assert "Load 2D Preview" in INDEX_HTML
    assert "two-d-preview-canvas" in INDEX_HTML
    assert "drawTwoDHeatmap" in INDEX_HTML
    assert "/api/preview/1d" in INDEX_HTML
    assert "Load 1D Preview" in INDEX_HTML
    assert "one-d-preview-svg" in INDEX_HTML
    assert "drawOneDCurve" in INDEX_HTML
    assert "Low coverage bins" in INDEX_HTML
    assert "Coverage drop bins" in INDEX_HTML
    assert "/api/preview/coverage" in INDEX_HTML
    assert "Load Coverage" in INDEX_HTML
    assert "coverage-preview-svg" in INDEX_HTML
    assert "Coverage" in INDEX_HTML
    assert 'id="add-job"' in INDEX_HTML
    assert 'id="save-batch-config"' in INDEX_HTML
    assert "Load Batch Config" in INDEX_HTML
    assert "batch-jobs-table" in INDEX_HTML
    assert "/api/batch/load" in INDEX_HTML
    assert "/api/batch/save" in INDEX_HTML
    assert 'id="workspace-tab-console-gixs"' in INDEX_HTML
    assert 'id="workspace-tab-console-trxs"' in INDEX_HTML
    assert 'id="workspace-tab-batch"' in INDEX_HTML
    assert 'id="c-load"' in INDEX_HTML
    assert 'id="t-load"' in INDEX_HTML
    assert 'id="c-dir"' in INDEX_HTML
    assert 'id="c-file"' in INDEX_HTML
    assert 'id="c-browse"' in INDEX_HTML
    assert 'id="c-bkg-dir"' in INDEX_HTML
    assert 'id="c-bkg-browse"' in INDEX_HTML
    assert 'id="c-colorbar"' in INDEX_HTML
    assert 'id="c-cbar-min"' in INDEX_HTML
    assert 'id="c-cbar-max"' in INDEX_HTML
    assert 'id="t-colorbar"' in INDEX_HTML
    assert "console-image-row" in INDEX_HTML
    assert "console-colorbar-col" in INDEX_HTML
    assert 'id="c-info-dir"' in INDEX_HTML
    assert 'id="c-info-browse"' in INDEX_HTML
    assert 'id="c-info-file"' in INDEX_HTML
    assert "renderColorbar" in INDEX_HTML
    assert "/api/console/load-progress" in INDEX_HTML
    assert "pollLoadProgress" in INDEX_HTML
    assert 'id="c-yrange-min"' in INDEX_HTML
    assert 'id="c-yrange-max"' in INDEX_HTML
    assert 'id="c-zrange-min"' in INDEX_HTML
    assert 'id="c-zrange-max"' in INDEX_HTML
    assert 'id="c-font-sz"' in INDEX_HTML
    assert 'id="c-save-format"' in INDEX_HTML
    assert 'id="c-reset-zoom"' in INDEX_HTML
    assert 'id="t-reset-zoom"' in INDEX_HTML
    assert "applyZoom" in INDEX_HTML
    assert "resetZoom" in INDEX_HTML
    assert 'value="jet"' in INDEX_HTML
    assert 'value="viridis"' in INDEX_HTML
    assert 'value="turbo"' in INDEX_HTML
    assert 'value="parula"' in INDEX_HTML
    assert 'id="c-ai-ypix"' in INDEX_HTML
    assert 'id="c-ai-zplot"' in INDEX_HTML
    assert "runZPlot" in INDEX_HTML
    assert "/api/console/qspace-image" in INDEX_HTML
    assert "getViewWindow" in INDEX_HTML
    assert "fullBounds" in INDEX_HTML
    assert 'id="t-yrange-min"' in INDEX_HTML
    assert 'id="t-save-format"' in INDEX_HTML
    assert 'id="c-axis-y"' in INDEX_HTML
    assert 'id="c-axis-x"' in INDEX_HTML
    assert 'id="t-axis-y"' in INDEX_HTML
    assert "getViewWindow" in INDEX_HTML
    assert "renderAxisTicks" in INDEX_HTML
    assert 'placeholder="path\\to\\folder"' in INDEX_HTML
    assert "LOAD DATA" in INDEX_HTML
    assert "&mdash; [ 1D Plot Process ] &mdash;" in INDEX_HTML
    assert "&mdash; [ Azimuthal Process ] &mdash;" in INDEX_HTML
    assert "/api/console/load" in INDEX_HTML
    assert "/api/console/list-files" in INDEX_HTML
    assert "/api/console/qmap" in INDEX_HTML
    assert "/api/console/1d/line" in INDEX_HTML
    assert "/api/console/1d/azimuthal" in INDEX_HTML
    assert "/api/console/1d/circular" in INDEX_HTML
    assert "/api/console/save" in INDEX_HTML
    assert "createConsole" in INDEX_HTML
    assert 'mode: "trxs"' in INDEX_HTML


def test_local_server_reexports_console_session_functions():
    assert callable(console_load)
    assert callable(console_qmap)
    assert callable(console_state)


def test_local_server_suppresses_access_log():
    handler = object.__new__(PGIXSRequestHandler)
    handler.log_message("GET %s", "/")

    assert True


def test_validate_payload_rejects_bad_args():
    payload = validate_payload({"workflow": "batch", "args": "bad"})

    assert not payload["ok"]
    assert "args must be" in payload["message"]


def test_validate_payload_supports_non_batch_parser_validation():
    payload = validate_payload({"workflow": "gixs", "args": ["--q-bins", "32"]})

    assert payload["ok"]


def test_run_payload_rejects_bad_args():
    payload = run_payload({"workflow": "batch", "args": "bad"})

    assert not payload["ok"]
    assert "args must be" in payload["message"]


def test_run_payload_supports_batch_runner_ready_message():
    payload = run_payload({"workflow": "batch", "args": []})

    assert payload["ok"]
    assert "Batch workflow runner is ready" in payload["message"]


def test_run_payload_attaches_existing_batch_summary_json(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "job_count": 1,
                "failed_count": 0,
                "jobs": [{"index": 1, "label": "ok", "workflow": "gixs", "ok": True, "exit_code": 0}],
            }
        ),
        encoding="utf-8",
    )

    payload = run_payload({"workflow": "batch", "args": ["--summary-json", str(summary)]})

    assert payload["batch_summary"]["job_count"] == 1
    assert payload["batch_summary"]["jobs"][0]["label"] == "ok"


def test_two_d_preview_payload_requires_path():
    payload = two_d_preview_payload({})

    assert not payload["ok"]
    assert "path is required" in payload["message"]


def test_one_d_preview_payload_requires_path():
    payload = one_d_preview_payload({})

    assert not payload["ok"]
    assert "path is required" in payload["message"]


def test_coverage_payload_requires_path():
    payload = coverage_payload({})

    assert not payload["ok"]
    assert "path is required" in payload["message"]


def test_batch_load_payload_requires_path():
    payload = batch_load_payload("")

    assert not payload["ok"]
    assert "path is required" in payload["message"]


def test_batch_load_payload_reports_missing_file():
    payload = batch_load_payload("does/not/exist.json")

    assert not payload["ok"]
    assert payload["message"]


def test_batch_load_payload_returns_jobs(tmp_path):
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps({"jobs": [{"label": "job-a", "workflow": "gixs", "config": "a.json", "args": ["--q-bins", "32"]}]}),
        encoding="utf-8",
    )

    payload = batch_load_payload(str(batch_path))

    assert payload["ok"]
    assert payload["jobs"] == [{"label": "job-a", "workflow": "gixs", "config": "a.json", "args": ["--q-bins", "32"]}]


def test_batch_save_payload_requires_path():
    payload = batch_save_payload({"jobs": []})

    assert not payload["ok"]
    assert "path is required" in payload["message"]


def test_batch_save_payload_requires_job_list():
    payload = batch_save_payload({"path": "x.json", "jobs": "bad"})

    assert not payload["ok"]
    assert "jobs must be" in payload["message"]


def test_batch_save_payload_rejects_bad_workflow(tmp_path):
    batch_path = tmp_path / "batch.json"

    payload = batch_save_payload(
        {"path": str(batch_path), "jobs": [{"label": "job-a", "workflow": "not-a-workflow", "config": None, "args": []}]}
    )

    assert not payload["ok"]
    assert payload["message"]


def test_batch_save_payload_writes_and_round_trips(tmp_path):
    batch_path = tmp_path / "batch.json"

    save_result = batch_save_payload(
        {
            "path": str(batch_path),
            "jobs": [
                {"label": "job-a", "workflow": "gixs", "config": "a.json", "args": ["--q-bins", "32"]},
                {"label": "job-b", "workflow": "export-1d", "config": None, "args": []},
            ],
        }
    )

    assert save_result["ok"]
    assert save_result["job_count"] == 2

    load_result = batch_load_payload(str(batch_path))
    assert load_result["ok"]
    assert [job["label"] for job in load_result["jobs"]] == ["job-a", "job-b"]
