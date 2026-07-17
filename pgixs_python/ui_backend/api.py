"""JSON-serializable API payload helpers for the local web UI."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from pgixs_python.ui_backend.discovery import get_backend_status, list_workflows


def status_payload() -> dict[str, Any]:
    """Return backend status as a JSON-serializable dictionary."""

    return to_jsonable(get_backend_status())


def workflows_payload() -> dict[str, Any]:
    """Return workflow metadata/schema payload for UI startup."""

    return {"workflows": [to_jsonable(info) for info in list_workflows()]}


def startup_payload() -> dict[str, Any]:
    """Return all first-load UI metadata in one payload."""

    return {
        "status": status_payload(),
        "workflows": workflows_payload()["workflows"],
    }


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and tuples into JSON-friendly containers."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

