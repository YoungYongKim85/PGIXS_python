import argparse
import json

from pgixs_python.ui_backend import UIWorkflowRequest, build_workflow_argv, run_workflow, validate_workflow


def test_build_workflow_argv_uses_config_then_args():
    request = UIWorkflowRequest(workflow="gixs", config="config.json", args=["--q-bins", "64"])

    assert build_workflow_argv(request) == ["--config", "config.json", "--q-bins", "64"]


def test_validate_workflow_accepts_valid_config(tmp_path):
    config = tmp_path / "gixs.json"
    config.write_text(json.dumps({"q_bins": 64, "q_scale": "linear"}), encoding="utf-8")

    result = validate_workflow(UIWorkflowRequest(workflow="gixs", config=config))

    assert result.ok
    assert result.message == "validation ok"


def test_validate_workflow_rejects_unknown_config_key(tmp_path):
    config = tmp_path / "gixs.json"
    config.write_text(json.dumps({"q_bnis": 64}), encoding="utf-8")

    result = validate_workflow(UIWorkflowRequest(workflow="gixs", config=config))

    assert not result.ok
    assert result.exit_code == 2
    assert "Unknown config option" in result.message


def test_run_workflow_captures_runner_output():
    def runner(argv):
        assert argv == ["--q-bins", "64"]
        print("done")
        return 0

    result = run_workflow(
        UIWorkflowRequest(workflow="trxs", args=["--q-bins", "64"]),
        runners={"trxs": runner, "gixs": runner, "export-1d": runner},
    )

    assert result.ok
    assert result.message == "done"
    assert result.stdout.strip() == "done"


def test_validate_workflow_supports_injected_parser():
    def build_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=False)
        parser.add_argument("--value", type=int, default=1)
        return parser

    result = validate_workflow(
        UIWorkflowRequest(workflow="export-1d", args=["--value", "4"]),
        parsers={"trxs": build_parser, "gixs": build_parser, "export-1d": build_parser},
    )

    assert result.ok


def test_validate_workflow_reports_unsupported_workflow():
    result = validate_workflow(UIWorkflowRequest(workflow="unknown"))

    assert not result.ok
    assert "Unsupported workflow" in result.message
