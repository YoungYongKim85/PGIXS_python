"""JSON Schema helpers for PGIXS workflow configuration files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_config_schema(parser: argparse.ArgumentParser, title: str) -> dict[str, Any]:
    """Build a JSON Schema from an argparse workflow parser."""

    properties: dict[str, dict[str, Any]] = {}
    for action in parser._actions:  # noqa: SLF001 - argparse has no public action iterator.
        if not action.dest or action.dest == "help":
            continue
        properties[action.dest] = _schema_for_action(action)

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }


def build_workflow_schema(workflow: str) -> dict[str, Any]:
    """Return the config schema for a named workflow."""

    workflow_key = workflow.lower().replace("_", "-")
    if workflow_key == "trxs":
        from pgixs_python.workflows.trxs_process import build_parser

        return build_config_schema(build_parser(), "PGIXS TrXS Workflow Config")
    if workflow_key == "gixs":
        from pgixs_python.workflows.gixs_process import build_parser

        return build_config_schema(build_parser(), "PGIXS GIXS Workflow Config")
    if workflow_key in {"export-1d", "export_1d"}:
        from pgixs_python.workflows.export_1d_from_h5 import build_parser

        return build_config_schema(build_parser(), "PGIXS 1D H5 Export Config")
    raise ValueError(f"Unknown workflow for config schema: {workflow}")


def write_workflow_schema(workflow: str, output_path: str | Path) -> Path:
    """Write one workflow config schema to a JSON file."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = build_workflow_schema(workflow)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PGIXS workflow JSON config schema files.")
    parser.add_argument("--workflow", choices=["trxs", "gixs", "export-1d", "all"], default="all")
    parser.add_argument("--out", default="docs", help="Output file for one schema or directory for --workflow all.")
    args = parser.parse_args(argv)

    if args.workflow == "all":
        out_dir = Path(args.out)
        written = [
            write_workflow_schema("trxs", out_dir / "trxs_workflow_config_schema.json"),
            write_workflow_schema("gixs", out_dir / "gixs_workflow_config_schema.json"),
            write_workflow_schema("export-1d", out_dir / "export_1d_config_schema.json"),
        ]
    else:
        written = [write_workflow_schema(args.workflow, args.out)]

    for path in written:
        print(path)
    return 0


def _schema_for_action(action: argparse.Action) -> dict[str, Any]:
    schema: dict[str, Any] = {"description": action.help or ""}
    if action.choices is not None:
        schema["enum"] = list(action.choices)
        schema["type"] = _choice_type(action.choices)
    elif isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001 - no public bool flag class.
        schema["type"] = "boolean"
        schema["default"] = bool(action.default)
    elif action.type is int:
        schema["type"] = ["integer", "null"] if action.default is None else "integer"
    elif action.type is float:
        schema["type"] = ["number", "null"] if action.default is None else "number"
    else:
        schema["type"] = ["string", "null"] if action.required is False else "string"

    if action.default is not None and action.default is not argparse.SUPPRESS:
        schema["default"] = action.default
    _apply_known_constraints(action.dest, schema)
    return schema


def _choice_type(choices: Any) -> str:
    values = list(choices)
    if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "number"
    return "string"


def _apply_known_constraints(dest: str, schema: dict[str, Any]) -> None:
    if dest in {"q_bins"}:
        schema["minimum"] = 1
    if dest in {"retry_count", "frame_index"}:
        schema["minimum"] = 0
    if dest == "config":
        schema["readOnly"] = True


__all__ = [
    "build_config_schema",
    "build_workflow_schema",
    "write_workflow_schema",
]


if __name__ == "__main__":
    raise SystemExit(main())
