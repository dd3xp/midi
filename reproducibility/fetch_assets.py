"""Download versioned assets, verify SHA256, and extract without overwriting."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'assets.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', nargs='+', choices=list(manifest))
    args = parser.parse_args()
    base = ROOT / 'assets'
    base.mkdir(exist_ok=True)
    for name in args.assets:
        item = manifest[name]
        dest = base / name
        if dest.exists():
            raise SystemExit(f'{dest} already exists; choose a fresh checkout to re-download.')
        archive = base / item['file']
        if not archive.exists():
            partial = archive.with_suffix('.zip.part')
            print(f'Downloading {name}: {item["bytes"]:,} bytes', flush=True)
            req = urllib.request.Request(item['url'], headers={'User-Agent': 'midi-reproducibility'})
            with urllib.request.urlopen(req, timeout=120) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            partial.replace(archive)
        digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
        if archive.stat().st_size != item['bytes'] or digest != item['sha256']:
            raise SystemExit(f'Integrity check failed: {archive}')
        with tempfile.TemporaryDirectory(prefix='extract-', dir=base) as temporary:
            root = Path(temporary).resolve()
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    target = (root / member.filename).resolve()
                    if not target.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError(f'Unsafe archive entry: {member.filename}')
                z.extractall(root)
            root.rename(dest)
        print(f'Verified and extracted: {dest}', flush=True)

if __name__ == '__main__':
    main()
