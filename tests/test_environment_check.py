import importlib.util
import sys
from pathlib import Path


TOOLS_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", TOOLS_PATH)
check_environment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_environment
SPEC.loader.exec_module(check_environment)


def test_read_pinned_requirements_ignores_comments_and_index(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                "# comment",
                "--extra-index-url https://example.invalid",
                "numpy==1.23.5",
                "editable-package",
                "h5py==3.10.0",
            ]
        ),
        encoding="utf-8",
    )

    assert check_environment.read_pinned_requirements(req) == [("numpy", "1.23.5"), ("h5py", "3.10.0")]


def test_check_requirement_reports_missing_package():
    result = check_environment.check_requirement("definitely-not-a-real-pgixs-package", "0.0.1")

    assert result.installed is None
    assert not result.ok
