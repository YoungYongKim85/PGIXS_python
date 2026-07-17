from pgixs_python.ui_backend import get_backend_status, get_workflow_info, list_workflows


def test_list_workflows_returns_schema_metadata():
    workflows = list_workflows()

    assert [item.workflow for item in workflows] == ["gixs", "trxs", "export-1d"]
    assert all(item.schema["type"] == "object" for item in workflows)
    assert "q_bins" in workflows[0].schema["properties"]


def test_get_workflow_info_normalizes_name():
    info = get_workflow_info("export_1d")

    assert info.workflow == "export-1d"
    assert "Export" in info.title


def test_get_backend_status_contains_runtime_contract():
    status = get_backend_status()

    assert status.version
    assert status.python_version
    assert "gixs" in status.workflows
    assert status.root

