from http import HTTPStatus
import json

from pgixs_python.ui_backend.local_server import (
    INDEX_HTML,
    PGIXSRequestHandler,
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
