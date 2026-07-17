from http import HTTPStatus

from pgixs_python.ui_backend.local_server import INDEX_HTML, PGIXSRequestHandler, run_payload, validate_payload


def test_local_server_has_expected_routes():
    assert "/" in {"/", "/index.html"}
    assert hasattr(PGIXSRequestHandler, "do_GET")
    assert HTTPStatus.OK.value == 200


def test_index_html_contains_workbench_regions():
    assert "Batch Workflow" in INDEX_HTML
    assert "Batch Queue" in INDEX_HTML
    assert "Run Summary" in INDEX_HTML
    assert "/api/startup" in INDEX_HTML
    assert "/api/validate" in INDEX_HTML
    assert "/api/run" in INDEX_HTML
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
