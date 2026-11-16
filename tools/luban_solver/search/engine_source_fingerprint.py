#!/usr/bin/env python3
"""Deterministic source fingerprint shared by .NET search and Unity runtime builds."""
import argparse, hashlib, pathlib

def fingerprint(root):
    root=pathlib.Path(root).resolve(); h=hashlib.sha256()
    files=sorted((p.relative_to(root).as_posix(),p) for p in root.rglob('*') if p.is_file() and p.suffix in {'.cs','.csproj','.props'} and not {'bin','obj'}.intersection(p.relative_to(root).parts))
    if not files:raise ValueError('No engine C# source files found')
    for name,path in files:
        h.update(name.encode('utf-8'));h.update(b'\0');h.update(path.read_bytes());h.update(b'\0')
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--csharp-output');a=p.parse_args()
    value=fingerprint(a.root)
    if a.csharp_output:
        out=pathlib.Path(a.csharp_output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text('internal static class EngineSourceFingerprint { public const string Value = "'+value+'"; }\n')
    print(value)
