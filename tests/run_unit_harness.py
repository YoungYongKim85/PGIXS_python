from __future__ import annotations

import importlib
import inspect
import io
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class TestResult:
    name: str
    status: str
    message: str = ""


def main() -> int:
    results = run_all_tests()
    passed = sum(1 for result in results if result.status == "PASS")
    skipped = sum(1 for result in results if result.status == "SKIP")
    failed = [result for result in results if result.status == "FAIL"]
    for result in results:
        if result.status != "PASS":
            print(f"{result.status}: {result.name} {result.message}".rstrip())
    print(f"PGIXS unit harness: passed={passed}, skipped={skipped}, failed={len(failed)}")
    return 1 if failed else 0


def run_all_tests() -> list[TestResult]:
    results: list[TestResult] = []
    for module_path in sorted((ROOT / "tests").glob("test_*.py")):
        module_name = f"tests.{module_path.stem}"
        module = importlib.import_module(module_name)
        for test_name, test_func in sorted(_iter_test_functions(module)):
            results.append(_run_one_test(module_name, test_name, test_func))
    return results


def _iter_test_functions(module):
    for name, value in vars(module).items():
        if name.startswith("test_") and callable(value):
            yield name, value


def _run_one_test(module_name: str, test_name: str, test_func) -> TestResult:
    full_name = f"{module_name}.{test_name}"
    signature = inspect.signature(test_func)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            if not signature.parameters:
                test_func()
            elif list(signature.parameters) == ["tmp_path"]:
                with tempfile.TemporaryDirectory() as temp_dir:
                    test_func(Path(temp_dir))
            else:
                return TestResult(full_name, "SKIP", f"unsupported parameters: {list(signature.parameters)}")
    except ModuleNotFoundError as exc:
        if exc.name == "pytest":
            return TestResult(full_name, "SKIP", "optional dependency skip requested pytest")
        return TestResult(full_name, "FAIL", _format_exception(stdout.getvalue(), stderr.getvalue()))
    except BaseException:  # noqa: BLE001 - this harness must report failures without pytest.
        return TestResult(full_name, "FAIL", _format_exception(stdout.getvalue(), stderr.getvalue()))
    return TestResult(full_name, "PASS")


def _format_exception(stdout: str = "", stderr: str = "") -> str:
    parts = [traceback.format_exc(limit=8).strip()]
    if stdout.strip():
        parts.append("[stdout]\n" + stdout.strip())
    if stderr.strip():
        parts.append("[stderr]\n" + stderr.strip())
    return "\n".join(parts).replace("\n", "\n    ")


if __name__ == "__main__":
    raise SystemExit(main())
