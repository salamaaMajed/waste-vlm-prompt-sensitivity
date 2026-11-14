#!/usr/bin/env python3
"""Restore losslessly compressed evidence; verify hashes before replacing files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile


def restore(root, verify_only=False):
    root = Path(root).resolve()
    entries = json.loads((root / 'measurements/COMPRESSED_FILES.json').read_text())
    for entry in entries:
        source = (root / entry['archive']).resolve()
        target = (root / entry['path']).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root):
            raise ValueError('Evidence path escapes repository')
        if target.exists():
            existing = hashlib.sha256()
            with target.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    existing.update(chunk)
            if existing.hexdigest() != entry['sha256']:
                raise ValueError(f'Refusing to overwrite modified evidence: {entry["path"]}')
        temporary = None
        try:
            if not verify_only and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
            digest = hashlib.sha256()
            size = 0
            with gzip.open(source, 'rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    if temporary is not None:
                        temporary.write(chunk)
            if digest.hexdigest() != entry['sha256'] or size != entry['size_bytes']:
                raise ValueError(f'Corrupt evidence: {entry["archive"]}')
            if temporary is not None:
                temporary.close()
                Path(temporary.name).replace(target)
            print(('Verified ' if verify_only else 'Ready ') + entry['path'])
        finally:
            if temporary is not None:
                temporary.close()
                Path(temporary.name).unlink(missing_ok=True)
    print(f'{len(entries)} evidence files verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    restore(Path(__file__).resolve().parents[1], args.verify_only)
