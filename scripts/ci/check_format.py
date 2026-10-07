"""Check all Python files except byte-identical, explicitly recorded historical debt."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def checked_files(root: Path, debt: dict[str, str]) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for directory in ("src", "tests", "migrations", "scripts")
        for path in (root / directory).rglob("*.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != debt.get(str(path.relative_to(root)))
    )


def main() -> int:
    root = Path.cwd()
    debt = json.loads((root / "ci/format-debt.json").read_text())["files"]
    files = checked_files(root, debt)
    print(f"Checking {len(files)} Python files; unchanged historical exemptions: {len(debt)}")
    return subprocess.call([sys.executable, "-m", "ruff", "format", "--check", *files])


if __name__ == "__main__":
    raise SystemExit(main())
