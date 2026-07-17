"""Backend helpers for the future PGIXS Tab1 local web UI."""

from pgixs_python.ui_backend.discovery import (
    UIBackendStatus,
    UIWorkflowInfo,
    get_backend_status,
    get_workflow_info,
    list_workflows,
)
from pgixs_python.ui_backend.service import (
    UIWorkflowRequest,
    UIWorkflowResult,
    build_workflow_argv,
    run_workflow,
    validate_workflow,
)

__all__ = [
    "UIBackendStatus",
    "UIWorkflowInfo",
    "UIWorkflowRequest",
    "UIWorkflowResult",
    "build_workflow_argv",
    "get_backend_status",
    "get_workflow_info",
    "list_workflows",
    "run_workflow",
    "validate_workflow",
]
