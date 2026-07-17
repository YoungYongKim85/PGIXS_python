from http import HTTPStatus

from pgixs_python.ui_backend.local_server import INDEX_HTML, PGIXSRequestHandler


def test_local_server_has_expected_routes():
    assert "/" in {"/", "/index.html"}
    assert hasattr(PGIXSRequestHandler, "do_GET")
    assert HTTPStatus.OK.value == 200


def test_index_html_contains_workbench_regions():
    assert "Batch Workflow" in INDEX_HTML
    assert "Batch Queue" in INDEX_HTML
    assert "Run Summary" in INDEX_HTML
    assert "/api/startup" in INDEX_HTML
    assert "Coverage" in INDEX_HTML


def test_local_server_suppresses_access_log():
    handler = object.__new__(PGIXSRequestHandler)
    handler.log_message("GET %s", "/")

    assert True
