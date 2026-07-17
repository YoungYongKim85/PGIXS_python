"""Check whether the active Python environment matches PGIXS Tab1 needs."""

from __future__ import annotations

import argparse
import compileall
import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
TORCH_REQUIREMENTS = ROOT / "requirements-torch-cu113.txt"


@dataclass
class RequirementCheck:
    package: str
    expected: str
    installed: str | None
    ok: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the active PGIXS Python environment.")
    parser.add_argument("--include-torch", action="store_true", help="Also check requirements-torch-cu113.txt.")
    parser.add_argument("--run-tests", action="store_true", help="Run smoke test, unit harness, and compileall.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = check_requirements(REQUIREMENTS)
    if args.include_torch:
        checks.extend(check_requirements(TORCH_REQUIREMENTS))

    python_ok = sys.version_info[:2] == (3, 10)
    print(f"Python: {sys.version.split()[0]} ({'OK' if python_ok else 'EXPECTED 3.10.x'})")
    for check in checks:
        installed = check.installed if check.installed is not None else "not installed"
        status = "OK" if check.ok else "MISMATCH"
        print(f"{check.package}: expected {check.expected}, installed {installed} [{status}]")

    ok = python_ok and all(check.ok for check in checks)
    if args.run_tests:
        ok = run_validation_commands() and ok
    return 0 if ok else 1


def check_requirements(path: Path) -> list[RequirementCheck]:
    return [check_requirement(package, version) for package, version in read_pinned_requirements(path)]


def read_pinned_requirements(path: Path) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if "==" not in line:
            continue
        package, version = line.split("==", 1)
        requirements.append((package.strip(), version.strip()))
    return requirements


def check_requirement(package: str, expected: str) -> RequirementCheck:
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    return RequirementCheck(package=package, expected=expected, installed=installed, ok=(installed == expected))


def run_validation_commands() -> bool:
    commands = [
        [sys.executable, "tests/run_smoke.py"],
        [sys.executable, "tests/run_unit_harness.py"],
    ]
    ok = True
    for command in commands:
        print(f"Running: {' '.join(command)}")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        ok = ok and completed.returncode == 0
    print("Running: compileall pgixs_python tests")
    ok = bool(compileall.compile_dir(ROOT / "pgixs_python", quiet=1)) and ok
    ok = bool(compileall.compile_dir(ROOT / "tests", quiet=1)) and ok
    return ok


if __name__ == "__main__":
    raise SystemExit(main())

