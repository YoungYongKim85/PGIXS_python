import json

from pgixs_python.workflows.config_schema import build_workflow_schema, write_workflow_schema
from pgixs_python.workflows.gixs_process import build_parser as build_gixs_parser
from pgixs_python.workflows.common import parser_config_keys


def test_build_workflow_schema_matches_gixs_parser_keys():
    schema = build_workflow_schema("gixs")

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == parser_config_keys(build_gixs_parser())
    assert schema["properties"]["frame_index"]["type"] == "integer"
    assert schema["properties"]["frame_index"]["minimum"] == 0
    assert schema["properties"]["q_bins"]["minimum"] == 1
    assert schema["properties"]["q_min"]["type"] == ["number", "null"]
    assert schema["properties"]["normalization"]["enum"] == ["None", "IC", "BC"]


def test_write_workflow_schema_writes_json(tmp_path):
    out = tmp_path / "schema.json"

    written = write_workflow_schema("export-1d", out)

    assert written == out
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["title"] == "PGIXS 1D H5 Export Config"
    assert "q_scale" in loaded["properties"]
