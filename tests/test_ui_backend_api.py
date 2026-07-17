from pgixs_python.ui_backend.api import startup_payload, status_payload, workflows_payload


def test_status_payload_is_json_ready():
    payload = status_payload()

    assert payload["version"]
    assert isinstance(payload["workflows"], list)


def test_workflows_payload_includes_schema_properties():
    payload = workflows_payload()

    workflows = payload["workflows"]
    assert workflows[0]["workflow"] == "gixs"
    assert "properties" in workflows[0]["schema"]


def test_startup_payload_combines_status_and_workflows():
    payload = startup_payload()

    assert "status" in payload
    assert "workflows" in payload
    assert payload["workflows"]

