"""Recover omitted original rules without changing any existing puzzle geometry.

Use the included, source-derived rule pack or verify against the extracted Unity
source with --source. Output is separate; shipped geometry is never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE_FIELDS = ("polyhedrons", "polyhedronTypes", "polyhedronBaseIndex", "directions")


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def fingerprint(value) -> str:
    data = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def base_data(value):
    return {key: value[key] for key in BASE_FIELDS}


def restore(source: Path | None, levels: Path, output: Path, rule_pack: Path | None = None):
    if source:
        raw_path = source / "Assets/Ass/Date/obj_datas.json"
        originals = json.loads(raw_path.read_text(encoding="utf-8-sig"))
        signatures = {}
        for index, puzzle in enumerate(originals):
            signatures.setdefault(fingerprint(base_data(puzzle)), []).append(index)
        source_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        original_count = len(originals)
    else:
        pack = json.loads((rule_pack or Path(__file__).with_name("original_rule_pack.json")).read_text())
        if pack["schema_version"] != 1 or sorted(row["level"] for row in pack["levels"]) != list(range(32)):
            raise ValueError("Invalid rule pack: need exactly one record for each of 32 levels")
        packed = {row["level"]: row for row in pack["levels"]}
        source_sha = pack["source_data_sha256"]
        original_count = pack["original_puzzles"]
    pending = []
    for level in range(32):
        current_path = levels / f"Puzzle_{level}.json"
        current = json.loads(current_path.read_text(encoding="utf-8-sig"))
        if source:
            source_current = json.loads(
                (source / f"Assets/Resources/Puzzles/Puzzle_{level}.json").read_text(encoding="utf-8-sig")
            )
            if normalize(current) != normalize(source_current):
                raise ValueError(f"Level {level}: uploaded source differs from installed level")
            candidates = signatures.get(fingerprint(base_data(current)), [])
            if len(candidates) != 1:
                raise ValueError(f"Level {level}: expected unique original, got {candidates}")
            original_id = candidates[0]
            rules = originals[original_id].get("moveRotates") or []
        else:
            row = packed[level]
            if fingerprint(current) != row["original_level_sha256"] or fingerprint(base_data(current)) != row["geometry_sha256"]:
                raise ValueError(f"Level {level}: source-derived rule pack does not match this puzzle")
            original_id, rules = row["original_index"], row["moveRotates"]
        restored = {**current, "moveRotates": rules}
        if base_data(restored) != base_data(current):
            raise AssertionError("Geometry changed during rule restoration")
        record = {
            "level": level,
            "original_index": original_id,
            "rule_count": len(rules),
            "geometry_sha256": fingerprint(base_data(current)),
            "input_file_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
            "restored_sha256": fingerprint(restored),
        }
        pending.append((level, restored, record))
    # Validate the entire mapping before writing any output.
    output = output.resolve()
    if output == levels.resolve() or (source and output == (source / "Assets/Resources/Puzzles").resolve()):
        raise ValueError("Output must be separate from both original level directories")
    output.mkdir(parents=True, exist_ok=True)
    for level, restored, _ in pending:
        (output / f"Puzzle_{level}.json").write_text(json.dumps(restored, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "source_data_sha256": source_sha,
        "original_puzzles": original_count,
        "restored_puzzles": len(pending),
        "geometry_unchanged": True,
        "note": "Conditional original actions, not complete solutions or validated playthroughs.",
        "levels": [row[2] for row in pending],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--source", type=Path, help="Cross-check the original extracted Unity project")
    sources.add_argument("--rule-pack", type=Path, help="Source-derived pack (defaults to the bundled pack)")
    parser.add_argument("--levels", type=Path, default=Path(__file__).parent / "levels")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = restore(args.source, args.levels, args.out, args.rule_pack)
    print(json.dumps({"levels": manifest["restored_puzzles"],
                      "rules": sum(row["rule_count"] for row in manifest["levels"]),
                      "geometry_unchanged": manifest["geometry_unchanged"]}))


if __name__ == "__main__":
    main()
