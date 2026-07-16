import json

from pgixs_python.workflows.config_schema import build_batch_schema, build_workflow_schema, write_workflow_schema
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


def test_build_batch_schema_accepts_object_or_array_shape():
    schema = build_batch_schema()

    assert schema["title"] == "PGIXS Batch Workflow Config"
    assert len(schema["oneOf"]) == 2
    job_props = schema["$defs"]["job"]["properties"]
    assert job_props["workflow"]["enum"] == ["trxs", "gixs", "export-1d"]
    assert job_props["args"]["items"]["type"] == "string"
