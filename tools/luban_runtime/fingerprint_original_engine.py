#!/usr/bin/env python3
"""Common source identity for standalone search and Unity-targeted engine builds."""
import hashlib
import json
from pathlib import Path
import sys


def source_fingerprint(folder):
    folder = Path(folder).resolve()
    selected = sorted(p for p in folder.rglob("*") if p.is_file()
                      and not any(x in {"bin", "obj"} for x in p.relative_to(folder).parts)
                      and p.suffix in {".cs", ".csproj", ".props"})
    digest = hashlib.sha256()
    files = {}
    for path in selected:
        relative = path.relative_to(folder).as_posix()
        content = path.read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode("utf-8") + b"\0" + content + b"\0")
    return {"algorithm": "sha256(sorted(relative_utf8+NUL+file_bytes+NUL))",
            "sha256": digest.hexdigest(), "files": files}


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "luban_solver/original_engine"
    print(json.dumps(source_fingerprint(folder), indent=2))
