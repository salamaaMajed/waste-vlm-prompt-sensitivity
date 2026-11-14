"""Check lossless restoration and protection of modified evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip
import hashlib
import json
import runpy


def test_restore_hashes_and_refuses_modified_evidence():
    script = Path(__file__).resolve().parents[1] / 'scripts/restore_measurements.py'
    restore = runpy.run_path(str(script))['restore']
    payload = b'path,label\nimage.jpg,plastic\n'
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'measurements').mkdir()
        archive = root / 'measurements/example.csv.gz'
        archive.write_bytes(gzip.compress(payload))
        manifest = [dict(path='measurements/example.csv', archive='measurements/example.csv.gz',
                         sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))]
        (root / 'measurements/COMPRESSED_FILES.json').write_text(json.dumps(manifest))
        restore(root, verify_only=True)
        target = root / 'measurements/example.csv'
        assert not target.exists()
        restore(root)
        assert target.read_bytes() == payload
        target.write_bytes(b'modified')
        try:
            restore(root)
        except ValueError:
            pass
        else:
            raise AssertionError('Modified evidence was overwritten')
        assert target.read_bytes() == b'modified'
