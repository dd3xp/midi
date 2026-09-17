from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
actual={f.relative_to(p).as_posix() for f in p.rglob('*') if f.is_file() and f!=p/'manifest.json' and '__pycache__' not in f.parts}
assert actual==set(m),{'missing':sorted(set(m)-actual),'extra':sorted(actual-set(m))}
for name,expected in m.items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print(f'Passed: all {len(m)} files match the manifest.')
