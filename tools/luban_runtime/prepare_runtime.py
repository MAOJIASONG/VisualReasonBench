#!/usr/bin/env python3
"""Create a NEW, self-contained Luban runtime with reproducible managed fixes.

Never patches the input runtime or an existing output. No game is launched here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from fingerprint_original_engine import source_fingerprint


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unmodified Luban directory")
    parser.add_argument("--output", type=Path, required=True, help="New runtime directory")
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--timeout-ms", type=int, default=300000)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--original-rules", type=Path, help="Enable restored-rule mode with these 32 rule JSON files")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        parser.error("--output must be new and outside --source")
    dll = Path("LuBanSuoEnv_Data/Managed/Assembly-CSharp.dll")
    binary = source / "LuBanSuoEnv.x86_64"
    if not binary.is_file() or not (source / dll).is_file():
        parser.error("--source is not a Linux Luban runtime")
    here = Path(__file__).resolve().parent
    if args.original_rules and args.skip_build:
        parser.error("Original-rule packages require a fresh build to verify the engine source fingerprint")
    engine_source = source_fingerprint(here.parent / "luban_solver/original_engine") if args.original_rules else None
    project = here / "patcher/RuntimePatcher.csproj"
    patcher = here / "patcher/bin/Release/net8.0/RuntimePatcher.dll"
    if not args.skip_build:
        subprocess.run([args.dotnet, "build", str(project), "-c", "Release"], check=True)
        if args.original_rules:
            subprocess.run([args.dotnet, "build", str(here / "extension/OriginalRuntimeExtension.csproj"),
                            "-c", "Release", "-p:UseUnity=true"], check=True)
            if source_fingerprint(here.parent / "luban_solver/original_engine") != engine_source:
                raise RuntimeError("Original engine source changed during build; prepare a new package from stable sources")
    source_hash = sha256(source / dll)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Staging is on the destination filesystem, so final publication is one rename.
    stage = Path(tempfile.mkdtemp(prefix=".luban-stage-", dir=output.parent))
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True)
        original_copy = stage / dll
        candidate = stage / dll.parent / "Assembly-CSharp.patched.dll"
        command = [args.dotnet, str(patcher), str(source / dll), str(candidate), str(args.timeout_ms)]
        if args.original_rules:
            rule_dir = args.original_rules.resolve()
            for level in range(32):
                if not (rule_dir / f"Puzzle_{level}.json").is_file():
                    raise ValueError(f"Missing original rule file for level {level}")
            extension_dir = here / "extension/bin/Release/netstandard2.1"
            command.append(str(extension_dir / "LubanOriginalRuntime.dll"))
            for name in ["LubanOriginalRuntime.dll", "OriginalEngine.dll"]:
                shutil.copy2(extension_dir / name, stage / dll.parent / name)
            rule_output = stage / "LuBanSuoEnv_Data/StreamingAssets/OriginalRules"
            rule_output.mkdir(parents=True, exist_ok=True)
            for level in range(32):
                shutil.copy2(rule_dir / f"Puzzle_{level}.json", rule_output)
        run = subprocess.run(command, check=True,
                             capture_output=True, text=True)
        metadata = json.loads(run.stdout)
        if sha256(source / dll) != source_hash:
            raise RuntimeError("Source changed during preparation")
        original_copy.unlink()
        candidate.rename(original_copy)
        metadata.update({"source_runtime": str(source), "source_commit":
                         "65538f43c98b13aeadaae95a57c5a9850c11ef79",
                         "prepare_script_sha256": sha256(__file__),
                         "patcher_source_sha256": sha256(here / "patcher/Program.cs"),
                         "command": ["<new-output-dll>" if value == str(candidate) else value for value in command],
                         "validation": "prepared_only_not_live_tested"})
        if args.original_rules:
            metadata["rule_file_sha256"] = {f"Puzzle_{i}.json": sha256(args.original_rules / f"Puzzle_{i}.json") for i in range(32)}
            metadata["extension_sha256"] = sha256(stage / dll.parent / "LubanOriginalRuntime.dll")
            metadata["original_engine_sha256"] = sha256(stage / dll.parent / "OriginalEngine.dll")
            if source_fingerprint(here.parent / "luban_solver/original_engine") != engine_source:
                raise RuntimeError("Original engine source changed during packaging")
            metadata["original_engine_source"] = engine_source
        (stage / "repair_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (stage / "LuBanSuoEnv.x86_64").chmod(binary.stat().st_mode)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"runtime": str(output), "manifest": str(output / "repair_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
